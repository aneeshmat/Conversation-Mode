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

os.environ["ORT_LOGGING_LEVEL"] = "3"

# ----------------- CONFIG -----------------
ONNX_PATH = "silero_vad.onnx"
VAD_SAMPLE_RATE = 16000
VAD_FRAME_16K = 512

MIC_ID = 4
LOOP_IDS = [1]

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
        self.vad_threshold = 0.30   # LOWERED FOR TESTING
        self.duck_vol = 20
        self.norm_vol = 80
        self.unduck_sec = 2.0


state = AppState()

# ----------------- VAD FOR YOUR MODEL -----------------

class SileroVADStateful:
    """
    Matches your ONNX model exactly:

    INPUTS:
      input  : [None, None] float
      state  : [2, None, 128] float
      sr     : [] int64

    OUTPUTS:
      output : [None, 1] float
      stateN : [None, None, None] float
    """

    def __init__(self, path):
        print("DEBUG: Loading ONNX model:", path)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1

        try:
            self.sess = ort.InferenceSession(
                path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
        except Exception:
            print("❌ Failed to load ONNX model!")
            import traceback
            traceback.print_exc()
            raise

        print("DEBUG: ONNX model loaded successfully.")

        self.name_x = "input"
        self.name_state = "state"
        self.name_sr = "sr"

        self.idx_prob = 0
        self.idx_stateN = 1

        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        print("DEBUG: VAD ready. State shape:", self.state.shape)

    def forward(self, audio):
        feed = {
            self.name_x: audio.reshape(1, -1).astype(np.float32),
            self.name_state: self.state,
            self.name_sr: np.array(16000, dtype=np.int64)
        }

        outs = self.sess.run(None, feed)

        prob = outs[self.idx_prob]
        self.state = outs[self.idx_stateN]

        return float(prob.reshape(-1)[0])


# ----------------- AUDIO / VOLUME -----------------

def set_volume(vol_percent):
    threading.Thread(
        target=lambda: subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{vol_percent}%"],
            capture_output=True
        ),
        daemon=True
    ).start()


def audio_loop():
    print("DEBUG: audio_loop() started")

    MIC_RATE = 48000
    LOOP_RATE = 48000

    BLOCK_MIC = int(VAD_FRAME_16K * (MIC_RATE / VAD_SAMPLE_RATE))
    BLOCK_LOOP = int(VAD_FRAME_16K * (LOOP_RATE / VAD_SAMPLE_RATE))

    try:
        vad = SileroVADStateful(ONNX_PATH)
    except Exception:
        print("❌ Could not initialize VAD. Exiting.")
        return

    loop_id = None
    for lid in LOOP_IDS:
        try:
            sd.query_devices(lid)
            loop_id = lid
            break
        except Exception:
            pass

    if loop_id is None:
        print("❌ No valid loopback device found.")
        return

    print("DEBUG: Opening audio streams…")

    try:
        m_in = sd.InputStream(device=MIC_ID, channels=1, samplerate=MIC_RATE, blocksize=BLOCK_MIC)
        l_in = sd.InputStream(device=loop_id, channels=1, samplerate=LOOP_RATE, blocksize=BLOCK_LOOP)

        m_in.start()
        l_in.start()

        print("DEBUG: Streams started successfully")
    except Exception:
        print("❌ Failed to open audio streams!")
        import traceback
        traceback.print_exc()
        return

    loop_history = deque(maxlen=40)
    last_speech = 0.0

    print("🚀 Audio loop running")

    while state.running:
        m_chunk, _ = m_in.read(BLOCK_MIC)
        l_chunk, _ = l_in.read(BLOCK_LOOP)

        m_raw = m_chunk[:, 0].astype(np.float32)
        l_raw = l_chunk[:, 0].astype(np.float32)

        rms = float(np.sqrt(np.mean(m_raw**2)))
        print(f"[MIC RMS] {rms:.4f}")

        t16 = np.linspace(0, 1, VAD_FRAME_16K, endpoint=False)
        m16 = np.interp(t16, np.linspace(0, 1, BLOCK_MIC, endpoint=False), m_raw)
        l16 = np.interp(t16, np.linspace(0, 1, BLOCK_LOOP, endpoint=False), l_raw)

        loop_history.append(l16)

        # TEMP: disable AEC for debugging
        clean = m16

        # Boost amplitude for VAD
        clean = clean * 3.0
        clean = np.clip(clean, -1, 1)

        state.prob = vad.forward(clean)
        print(f"[VAD PROB] {state.prob:.4f}")

        socketio.emit("stat", {"prob": round(state.prob, 2), "ducked": state.ducked})

        if state.enabled:
            if state.prob > state.vad_threshold:
                print(f"[VAD] Speech detected (prob={state.prob:.2f})")

                if not state.ducked:
                    print(f"[DUCK] Ducking volume → {state.duck_vol}%")
                    set_volume(state.duck_vol)
                    state.ducked = True

                last_speech = time.time()

            elif state.ducked and (time.time() - last_speech > state.unduck_sec):
                print(f"[RESTORE] Restoring volume → {state.norm_vol}%")
                set_volume(state.norm_vol)
                state.ducked = False


# ----------------- WEB -----------------

@app.route("/")
def index():
    return "<h1>VAD Ducking System Running</h1>"


@socketio.on("update")
def handle_update(data):
    val = float(data["val"]) if "." in str(data["val"]) else int(data["val"])
    setattr(state, data["key"], val)


if __name__ == "__main__":
    print("=== MAIN STARTED ===")
    t = threading.Thread(target=audio_loop, daemon=True)
    t.start()
    print("=== THREAD LAUNCHED ===")
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)