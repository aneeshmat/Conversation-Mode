import os
import time
import threading
import subprocess
from collections import deque
from flask import Flask, render_template, request
from flask_socketio import SocketIO

import numpy as np
import sounddevice as sd
import onnxruntime as ort

# --- CONFIG ---
ONNX_PATH = "silero_vad.onnx"
VAD_SAMPLE_RATE = 16000
VAD_FRAME_16K = 512
MIC_ID = 2
LOOP_IDS = [3, 4]

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

class AppState:
    def __init__(self):
        self.running = True
        self.enabled = True
        self.ducked = False
        self.prob = 0.0
        self.aec_delay = 14
        self.aec_strength = 0.85
        self.vad_threshold = 0.70
        self.duck_vol = 20
        self.norm_vol = 80
        self.unduck_sec = 2.0

state = AppState()

# --- AUDIO PROCESSING ---

def set_volume(vol):
    try:
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{int(vol)}%"], check=True)
    except: pass

class SileroVADStateful:
    def __init__(self, path):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inputs = {i.name: i for i in self.sess.get_inputs()}
        self.name_audio = next((n for n in inputs if "audio" in n or "input" in n or "x" in n), None)
        self.name_sr = next((n for n in inputs if "sr" in n or "sample_rate" in n), None)
        self.name_h = next((n for n in inputs if "h" in n and "n" not in n), None)
        self.name_c = next((n for n in inputs if "c" in n and "n" not in n), None)
        self.h, self.c = np.zeros((2, 1, 64), dtype=np.float32), np.zeros((2, 1, 64), dtype=np.float32)

    def forward(self, audio):
        feed = {self.name_audio: audio.reshape(1, -1).astype(np.float32)}
        if self.name_sr: feed[self.name_sr] = np.array([16000], dtype=np.int64)
        if self.name_h is not None: feed[self.name_h], feed[self.name_c] = self.h, self.c
        outs = self.sess.run(None, feed)
        out_names = [o.name for o in self.sess.get_outputs()]
        res = {out_names[i]: outs[i] for i in range(len(outs))}
        if self.name_h is not None:
            self.h = res[next(n for n in out_names if "h" in n and n != self.name_h)]
            self.c = res[next(n for n in out_names if "c" in n and n != self.name_c)]
        return float(res[next(n for n in out_names if "prob" in n or "output" in n or "y" in n)])

def audio_loop():
    SAMPLING_RATE = 48000
    loop_id = 3
    for lid in LOOP_IDS:
        try:
            sd.query_devices(lid); loop_id = lid; break
        except: continue

    vad = SileroVADStateful(ONNX_PATH)
    native_hop = int(VAD_FRAME_16K * (SAMPLING_RATE / 16000))
    loop_history = deque(maxlen=40)
    last_speech = 0.0

    with sd.InputStream(device=MIC_ID, channels=1, samplerate=SAMPLING_RATE, blocksize=native_hop) as m_in, \
         sd.InputStream(device=loop_id, channels=2, samplerate=SAMPLING_RATE, blocksize=native_hop) as l_in:
        while state.running:
            m_chunk, _ = m_in.read(native_hop)
            l_chunk, _ = l_in.read(native_hop)
            m16 = np.interp(np.linspace(0,1,512), np.linspace(0,1,native_hop), m_chunk[:,0]).astype(np.float32)
            l16 = np.interp(np.linspace(0,1,512), np.linspace(0,1,native_hop), l_chunk[:,0]).astype(np.float32)
            loop_history.append(l16)

            if len(loop_history) >= state.aec_delay:
                ref = list(loop_history)[-int(state.aec_delay)]
                clean = np.clip(m16 - (state.aec_strength * ref), -1.0, 1.0)
            else: clean = m16

            state.prob = vad.forward(clean)
            socketio.emit('stat', {'prob': round(state.prob, 2), 'ducked': state.ducked})

            if state.enabled:
                if state.prob > state.vad_threshold:
                    if not state.ducked: set_volume(state.duck_vol); state.ducked = True
                    last_speech = time.time()
                elif state.ducked and (time.time() - last_speech > state.unduck_sec):
                    set_volume(state.norm_vol); state.ducked = False

# --- WEB ROUTES ---

@app.route('/')
def index():
    return """
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
        <style>
            body { font-family: sans-serif; background: #121212; color: white; text-align: center; padding: 20px; }
            .card { background: #1e1e1e; padding: 20px; border-radius: 15px; margin-bottom: 20px; }
            input[type=range] { width: 100%; margin: 15px 0; }
            #status { font-size: 1.5em; font-weight: bold; padding: 10px; border-radius: 10px; }
            .active { background: #d32f2f; } .idle { background: #388e3c; }
            button { padding: 15px; width: 100%; border-radius: 10px; border: none; font-weight: bold; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="card">
            <div id="status" class="idle">MONITORING</div>
            <h2 id="prob_text">Prob: 0.00</h2>
        </div>
        <div class="card">
            <button id="toggleBtn" onclick="toggleSystem()" style="background: #d32f2f; color: white;">DISABLE DUCKING</button>
        </div>
        <div class="card">
            <label>AEC Delay (Bluetooth Lag)</label>
            <input type="range" min="1" max="35" step="1" value="14" oninput="update('aec_delay', this.value)">
            <label>VAD Threshold (Sensitivity)</label>
            <input type="range" min="0" max="1" step="0.05" value="0.70" oninput="update('vad_threshold', this.value)">
            <label>Normal Volume %</label>
            <input type="range" min="0" max="100" step="1" value="80" oninput="update('norm_vol', this.value)">
        </div>
        <script>
            var socket = io();
            var enabled = true;
            socket.on('stat', function(data) {
                document.getElementById('prob_text').innerText = "Prob: " + data.prob;
                let status = document.getElementById('status');
                if (data.ducked) { status.innerText = "DUCKING"; status.className = "active"; }
                else { status.innerText = "MONITORING"; status.className = "idle"; }
            });
            function update(key, val) { socket.emit('update', {key: key, val: val}); }
            function toggleSystem() {
                enabled = !enabled;
                socket.emit('update', {key: 'enabled', val: enabled});
                let btn = document.getElementById('toggleBtn');
                btn.innerText = enabled ? "DISABLE DUCKING" : "ENABLE DUCKING";
                btn.style.background = enabled ? "#d32f2f" : "#388e3c";
            }
        </script>
    </body>
    </html>
    """

@socketio.on('update')
def handle_update(data):
    setattr(state, data['key'], float(data['val']) if '.' in str(data['val']) else int(data['val']))

if __name__ == '__main__':
    threading.Thread(target=audio_loop, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000)