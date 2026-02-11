import os
import time
import threading
import subprocess
from collections import deque

from flask import Flask
from flask_socketio import SocketIO
import numpy as np
import sounddevice as sd
import onnxruntime as ort

# Suppress ONNXRuntime logs
os.environ["ORT_LOGGING_LEVEL"] = "3"

# ----------------- CONFIGURATION -----------------
ONNX_PATH = "silero_vad.onnx"
VAD_SAMPLE_RATE = 16000
VAD_FRAME_16K = 512
MIC_ID = 2
LOOP_IDS = [3, 4]  # candidate loopback devices (PulseAudio monitors, etc.)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


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

# ----------------- VAD ENGINE -----------------


class SileroVADStateful:
    def __init__(self, path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1

        self.sess = ort.InferenceSession(
            path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

        inputs = {i.name: i for i in self.sess.get_inputs()}
        self.name_x = next(
            (n for n in inputs if any(s in n.lower() for s in ["audio", "input", "x"])),
            "x",
        )
        self.name_sr = next((n for n in inputs if "sr" in n.lower()), None)
        self.name_h = next(
            (n for n in inputs if "h" in n.lower() and "n" not in n.lower()), "h"
        )
        self.name_c = next(
            (n for n in inputs if "c" in n.lower() and "n" not in n.lower()), "c"
        )

        outputs = {o.name: idx for idx, o in enumerate(self.sess.get_outputs())}
        self.idx_prob = outputs[
            next(
                n
                for n in outputs
                if any(k in n.lower() for k in ["prob", "output", "y"])
            )
        ]
        self.idx_h_out = outputs[
            next(n for n in outputs if "h" in n.lower() and "n" not in n.lower())
        ]
        self.idx_c_out = outputs[
            next(n for n in outputs if "c" in n.lower() and "n" not in n.lower())
        ]

        self.h = np.zeros((2, 1, 64), dtype=np.float32)
        self.c = np.zeros((2, 1, 64), dtype=np.float32)

    def forward(self, audio):
        feed = {
            self.name_x: audio.reshape(1, -1).astype(np.float32),
            self.name_h: self.h,
            self.name_c: self.c,
        }
        if self.name_sr:
            feed[self.name_sr] = np.array([VAD_SAMPLE_RATE], dtype=np.int64)

        outs = self.sess.run(None, feed)

        self.h = outs[self.idx_h_out]
        self.c = outs[self.idx_c_out]

        prob_val = outs[self.idx_prob]
        return float(np.array(prob_val).reshape(-1)[0])


# ----------------- AUDIO / VOLUME -----------------


def set_volume(vol_percent):
    def worker():
        try:
            subprocess.run(
                [
                    "pactl",
                    "set-sink-volume",
                    "@DEFAULT_SINK@",
                    f"{int(vol_percent)}%",
                ],
                check=True,
                capture_output=True,
            )
        except Exception:
            pass

    threading.Thread(target=worker, daemon=True).start()


def audio_loop():
    SAMPLING_RATE = VAD_SAMPLE_RATE  # 16 kHz to avoid resampling
    BLOCKSIZE = VAD_FRAME_16K        # 512 samples
    mic_channels = 1                 # Option C: force mono
    loop_channels = 1                # Option C: force mono

    loop_id = None
    for lid in LOOP_IDS:
        try:
            sd.query_devices(lid)
            loop_id = lid
            break
        except Exception:
            continue

    if loop_id is None:
        print("❌ No valid loopback device found from LOOP_IDS.")
        return

    try:
        vad = SileroVADStateful(ONNX_PATH)
    except Exception as e:
        print(f"Failed to load VAD: {e}")
        return

    loop_history = deque(maxlen=40)
    last_speech = 0.0

    print("🚀 Audio Started. Access GUI at http://10.6.1.47:5000")

    try:
        with sd.InputStream(
            device=MIC_ID,
            channels=mic_channels,
            samplerate=SAMPLING_RATE,
            blocksize=BLOCKSIZE,
        ) as m_in, sd.InputStream(
            device=loop_id,
            channels=loop_channels,
            samplerate=SAMPLING_RATE,
            blocksize=BLOCKSIZE,
        ) as l_in:
            while state.running:
                m_chunk, _ = m_in.read(BLOCKSIZE)
                l_chunk, _ = l_in.read(BLOCKSIZE)

                m16 = m_chunk[:, 0].astype(np.float32)
                l16 = l_chunk[:, 0].astype(np.float32)

                loop_history.append(l16)

                if len(loop_history) >= int(state.aec_delay):
                    ref = loop_history[-int(state.aec_delay)]
                    clean = np.clip(
                        m16 - (state.aec_strength * ref), -1.0, 1.0
                    )
                else:
                    clean = m16

                state.prob = vad.forward(clean)

                socketio.emit(
                    "stat",
                    {"prob": round(state.prob, 2), "ducked": state.ducked},
                )

                if state.enabled:
                    if state.prob > state.vad_threshold:
                        if not state.ducked:
                            set_volume(state.duck_vol)
                            state.ducked = True
                        last_speech = time.time()
                    elif state.ducked and (
                        time.time() - last_speech > state.unduck_sec
                    ):
                        set_volume(state.norm_vol)
                        state.ducked = False

    except Exception as e:
        print(f"❌ Audio Loop Error: {e}")


# ----------------- WEB ROUTES -----------------


@app.route("/")
def index():
    return """
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
        <style>
            body { font-family: sans-serif; background: #121212; color: white; text-align: center; padding: 20px; }
            .card { background: #1e1e1e; padding: 20px; border-radius: 15px; margin-bottom: 20px; }
            input[type=range] { width: 100%; margin: 15px 0; accent-color: #007bff; }
            #status { font-size: 1.5em; font-weight: bold; padding: 15px; border-radius: 10px; }
            .active { background: #d32f2f; } .idle { background: #388e3c; }
            button { padding: 15px; width: 100%; border-radius: 10px; border: none; font-size: 1.1em; font-weight: bold; color: white; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="card">
            <div id="status" class="idle">MONITORING</div>
            <h1 id="prob_text">Prob: 0.00</h1>
        </div>
        <div class="card">
            <button id="toggleBtn" onclick="toggleSystem()" style="background: #d32f2f;">DISABLE DUCKING</button>
        </div>
        <div class="card">
            <p>AEC Delay: <span id="aec_val">14</span></p>
            <input type="range" min="1" max="35" step="1" value="14"
                oninput="update('aec_delay', this.value); document.getElementById('aec_val').innerText=this.value">
            <p>VAD Sensitivity: <span id="vad_val">0.70</span></p>
            <input type="range" min="0" max="1" step="0.05" value="0.70"
                oninput="update('vad_threshold', this.value); document.getElementById('vad_val').innerText=this.value">
        </div>
        <script>
            var socket = io();
            var enabled = true;
            socket.on('stat', function(data) {
                document.getElementById('prob_text').innerText = "Prob: " + data.prob.toFixed(2);
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


@socketio.on("update")
def handle_update(data):
    val_str = str(data["val"])
    val = float(val_str) if "." in val_str else int(val_str)
    setattr(state, data["key"], val)


if __name__ == "__main__":
    threading.Thread(target=audio_loop, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)