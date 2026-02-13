"""
Conversation Mode - Linux (AEC + VAD + Relative Smooth Ducking + Tk GUI)

- Mic: JLAB TALK MICROPHONE (device 5)
- Ref: PipeWire monitor (device 12)
- I/O at 48 kHz (device-native), resample to 16 kHz for Silero VAD
- AEC runs at the device rate (48 kHz)
- Ducking:
    * Ducks to (current volume × DUCK_RATIO) with smooth fades when speech starts
    * STAYS ducked as long as speech continues
    * Restores (with smooth fade) only after DUCK_HOLD_SILENCE of no speech
    * If the user changes volume while ducked, we respect it
- VAD is gated so that frames dominated by speaker output do NOT trigger speech
"""

import os
import time
import threading
import subprocess
import ctypes
import queue
import re
import shutil

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk

# ---------------------------
# 1) SETTINGS
# ---------------------------
MIC_DEVICE_ID = 5
REF_DEVICE_ID = 12

DEVICE_RATE = 48000
SAMPLE_RATE = 16000

FRAME_SIZE = 3072           # 64 ms @ 48k
VAD_WINDOW_16K = 1024       # 64 ms @ 16k
VAD_THRESHOLD = 0.45

VAD_ATTACK = 0.60
VAD_RELEASE = 0.35
VAD_SMOOTHING = 0.6

GAIN_AFTER_AEC = 2.0
HP_ALPHA = 0.995

DUCK_RATIO = 0.5
FADE_STEPS = 6
FADE_STEP_MS = 30
VOLUME_RESTORE_TOL = 2
DUCK_HOLD_SILENCE = 1.3

AEC_DELAY = 240

MIC_DEVICE_ID = int(os.getenv("MIC_DEVICE_ID", MIC_DEVICE_ID))
REF_DEVICE_ID = int(os.getenv("REF_DEVICE_ID", REF_DEVICE_ID))

sd.default.latency = "high"

# ---------------------------
# 2) LOAD AEC CORE (.so)
# ---------------------------
aec_state = None
aec_lib = None

def load_aec_shared_object():
    global aec_lib, aec_state
    try:
        aec_lib = ctypes.CDLL("./aec_vad_v2.so")
        aec_lib.aec_process_buffer.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32),
            ctypes.c_int, ctypes.c_int
        ]
        aec_lib.aec_create.restype = ctypes.c_void_p
        aec_lib.aec_free.argtypes = [ctypes.c_void_p]
        aec_state = aec_lib.aec_create()
        print("✅ AEC Shared Object Loaded")
    except Exception as e:
        print(f"⚠️ AEC Core Failed: {e}")
        aec_state = None

load_aec_shared_object()

# ---------------------------
# 3) VOLUME CONTROL
# ---------------------------
def _parse_pactl_volume(stdout: str) -> int:
    vals = [int(m.group(1)) for m in re.finditer(r'(\d+)%', stdout)]
    if vals:
        return int(round(sum(vals) / len(vals)))
    return -1

def _pactl_get_volume() -> int:
    try:
        out = subprocess.run(
            ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
            capture_output=True, text=True, timeout=1.5
        )
        if out.returncode == 0:
            v = _parse_pactl_volume(out.stdout)
            if v >= 0:
                return v
    except Exception:
        pass
    return -1

def _pactl_set_volume(percent: int) -> bool:
    try:
        percent = max(0, min(150, int(percent)))
        out = subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"],
            capture_output=True, text=True, timeout=1.5
        )
        return out.returncode == 0
    except Exception:
        return False

_already_picked_control = None

def _amixer_try_control(control: str, percent: int = None) -> bool:
    try:
        if percent is None:
            out = subprocess.run(
                ["amixer", "get", control],
                capture_output=True, text=True, timeout=1.5
            )
            return out.returncode == 0 and ("[" in out.stdout)
        else:
            out = subprocess.run(
                ["amixer", "sset", control, f"{percent}%"],
                capture_output=True, text=True, timeout=1.5
            )
            return out.returncode == 0
    except Exception:
        return False

def _amixer_find_control():
    global _already_picked_control
    for name in ["Master", "PCM", "Speaker", "Headphone"]:
        if _amixer_try_control(name, None):
            _already_picked_control = name
            return
    _already_picked_control = None

def _amixer_get_volume() -> int:
    if _already_picked_control is None:
        _amixer_find_control()
    if not _already_picked_control:
        return -1
    try:
        out = subprocess.run(
            ["amixer", "get", _already_picked_control],
            capture_output=True, text=True, timeout=1.5
        )
        if out.returncode == 0:
            m = re.search(r'

\[(\d+)%\]

', out.stdout)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return -1

def _amixer_set_volume(percent: int) -> bool:
    if _already_picked_control is None:
        _amixer_find_control()
    if not _already_picked_control:
        return False
    try:
        percent = max(0, min(100, int(percent)))
        out = subprocess.run(
            ["amixer", "sset", _already_picked_control, f"{percent}%"],
            capture_output=True, text=True, timeout=1.5
        )
        return out.returncode == 0
    except Exception:
        return False

def _use_pactl() -> bool:
    return shutil.which("pactl") is not None

def get_current_volume_percent() -> int:
    if _use_pactl():
        v = _pactl_get_volume()
        if v >= 0:
            return v
    return _amixer_get_volume()

def set_current_volume_percent(percent: int) -> bool:
    if _use_pactl():
        return _pactl_set_volume(percent)
    return _amixer_set_volume(percent)

def smooth_set_volume(from_p: int, to_p: int,
                      steps: int = FADE_STEPS,
                      step_ms: int = FADE_STEP_MS):
    steps = max(1, int(steps))
    if steps == 1 or from_p == to_p:
        set_current_volume_percent(int(round(to_p)))
        return
    for t in np.linspace(from_p, to_p, steps):
        set_current_volume_percent(int(round(t)))
        time.sleep(step_ms / 1000.0)

# ---------------------------
# 4) VAD SETUP
# ---------------------------
USE_TORCH = True
model = None
vad_session = None

def setup_vad():
    global model, vad_session, USE_TORCH
    try:
        import torch
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        model.eval()
        USE_TORCH = True
        print("✅ Silero VAD (Torch) loaded")
    except Exception as e:
        print(f"⚠️ Torch/Silero VAD load failed: {e}\n→ Trying ONNX Runtime fallback...")
        try:
            import onnxruntime as ort
            import urllib.request
            import pathlib
            MODEL_PATH = pathlib.Path("silero_vad.onnx")
            if not MODEL_PATH.exists():
                print("Downloading Silero VAD ONNX model...")
                urllib.request.urlretrieve(
                    "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx",
                    MODEL_PATH
                )
            vad_session = ort.InferenceSession(str(MODEL_PATH),
                                               providers=["CPUExecutionProvider"])
            USE_TORCH = False
            print("✅ Silero VAD (ONNX) loaded")
        except Exception as e2:
            print(f"❌ ONNX fallback failed: {e2}")
            print("VAD unavailable. Speech detection will not work.")

setup_vad()

def warmup_vad():
    dummy = np.zeros(512, dtype=np.float32)
    _ = vad_prob_16k(dummy)

def _torch_vad_prob_16k(audio_16k: np.ndarray) -> float:
    import torch
    n = audio_16k.size
    if n >= 1024:
        seg = audio_16k[-1024:]
        wins = [seg[:512], seg[512:]]
        probs = []
        with torch.no_grad():
            for w in wins:
                t = torch.from_numpy(w.astype(np.float32))
                probs.append(float(model(t, SAMPLE_RATE).item()))
        return max(probs)
    elif n >= 512:
        w = audio_16k[-512:]
        with torch.no_grad():
            t = torch.from_numpy(w.astype(np.float32))
            return float(model(t, SAMPLE_RATE).item())
    else:
        return 0.0

def vad_prob_16k(audio_16k: np.ndarray) -> float:
    if USE_TORCH and model is not None:
        return _torch_vad_prob_16k(audio_16k)
    elif (not USE_TORCH) and vad_session is not None:
        inp_name = vad_session.get_inputs()[0].name
        x = audio_16k.astype(np.float32)[None, :]
        out = vad_session.run(None, {inp_name: x})[0]
        return float(out.ravel()[0])
    else:
        return 0.0

# ---------------------------
# 5) RESAMPLER + HIGHPASS
# ---------------------------
def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    src_n = len(x)
    if src_n == 0:
        return np.zeros(0, dtype=np.float32)
    dst_n = int(round(src_n * (dst_sr / src_sr)))
    if dst_n <= 1:
        return np.zeros(dst_n, dtype=np.float32)
    src_idx = np.linspace(0, src_n - 1, num=dst_n, dtype=np.float32)
    left = np.floor(src_idx).astype(np.int64)
    right = np.minimum(left + 1, src_n - 1)
    frac = src_idx - left
    y = (1.0 - frac) * x[left] + frac * x[right]
    return y.astype(np.float32)

def highpass_dc_block(x: np.ndarray, alpha: float = HP_ALPHA) -> np.ndarray:
    if x.size == 0:
        return x
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    for n in range(1, x.size):
        y[n] = x[n] - x[n-1] + alpha * y[n-1]
    return y

# ---------------------------
# 6) DUCK CONTROLLER
# ---------------------------
class DuckController:
    def __init__(self, ratio: float, hold_silence_sec: float):
        self.ratio = float(ratio)
        self.hold = float(hold_silence_sec)
        self.ducked = False
        self.baseline = None
        self.target = None
        self.last_voice_ts = 0.0

    def notify_speech(self):
        self.last_voice_ts = time.monotonic()
        if not self.ducked:
            base = get_current_volume_percent()
            if base < 0:
                return
            tgt = int(round(base * self.ratio))
            tgt = max(0, min(150, tgt))
            smooth_set_volume(base, tgt)
            self.baseline = base
            self.target = tgt
            self.ducked = True

    def update(self):
        if not self.ducked:
            return
        now = time.monotonic()
        if (now - self.last_voice_ts) >= self.hold:
            current = get_current_volume_percent()
            if current >= 0 and abs(current - self.target) <= VOLUME_RESTORE_TOL:
                smooth_set_volume(current, self.baseline)
            self.ducked = False
            self.baseline = None
            self.target = None

    def stop(self):
        if not self.ducked:
            return
        current = get_current_volume_percent()
        if (self.baseline is not None and current >= 0 and
            abs(current - self.target) <= VOLUME_RESTORE_TOL):
            smooth_set_volume(current, self.baseline)
        self.ducked = False
        self.baseline = None
        self.target = None

duck_ctrl = DuckController(DUCK_RATIO, DUCK_HOLD_SILENCE)

# ---------------------------
# 7) GLOBALS
# ---------------------------
last_vad_prob = 0.0
smoothed_vad = 0.0

ref_buffer = np.zeros(FRAME_SIZE, dtype=np.float32)
audio_q = queue.Queue(maxsize=16)
running_worker = False

# ---------------------------
# 8) AUDIO CALLBACKS
# ---------------------------
def ref_callback(indata, frames, time_info, status):
    if status:
        print("Ref status:", status)
    global ref_buffer
    ref = indata[:, 0].astype(np.float32)
    if len(ref) >= FRAME_SIZE:
        ref_buffer[:] = ref[:FRAME_SIZE]
    else:
        ref_buffer[:len(ref)] = ref
        ref_buffer[len(ref):] = 0.0

def mic_callback(indata, frames, time_info, status):
    if status:
        print("Mic status:", status)
    mic_frame = indata[:, 0].astype(np.float32).copy()
    ref_frame = ref_buffer.copy()
    try:
        audio_q.put_nowait((mic_frame, ref_frame))
    except queue.Full:
        try:
            _ = audio_q.get_nowait()
            audio_q.put_nowait((mic_frame, ref_frame))
        except Exception:
            pass

# ---------------------------
# 9) PROCESSING WORKER
# ---------------------------
def processing_worker():
    global last_vad_prob, smoothed_vad, running_worker
    while running_worker:
        try:
            mic_dev, ref_dev = audio_q.get(timeout=0.5)
        except queue.Empty:
            duck_ctrl.update()
            continue

        cleaned_dev = np.zeros_like(mic_dev, dtype=np.float32)
        if aec_state is not None and aec_lib is not None:
            aec_lib.aec_process_buffer(aec_state,
                                       ref_dev, mic_dev,
                                       cleaned_dev,
                                       FRAME_SIZE, AEC_DELAY)
        else:
            cleaned_dev[:] = mic_dev

        cleaned_dev = highpass_dc_block(cleaned_dev, alpha=HP_ALPHA)

        # --- NEW: energy-based gate to suppress VAD on pure speaker output ---
        mic_rms = float(np.sqrt(np.mean(cleaned_dev**2) + 1e-9))
        ref_rms = float(np.sqrt(np.mean(ref_dev**2) + 1e-9))
        ref_dominates = (ref_rms > 0.0 and mic_rms < 0.7 * ref_rms)

        cleaned_16k = resample_linear(cleaned_dev,
                                      src_sr=DEVICE_RATE,
                                      dst_sr=SAMPLE_RATE)

        if GAIN_AFTER_AEC != 1.0:
            cleaned_16k = cleaned_16k * np.float32(GAIN_AFTER_AEC)

        raw_prob = 0.0
        if cleaned_16k.size >= 512 and not ref_dominates:
            raw_prob = vad_prob_16k(cleaned_16k[-VAD_WINDOW_16K:])

        smoothed_vad = (VAD_SMOOTHING * smoothed_vad) + ((1.0 - VAD_SMOOTHING) * raw_prob)
        last_vad_prob = float(max(0.0, min(1.0, smoothed_vad)))

        if last_vad_prob >= VAD_ATTACK:
            duck_ctrl.notify_speech()

        duck_ctrl.update()

# ---------------------------
# 10) GUI
# ---------------------------
class DuckingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AEC + VAD (Linux)")
        self.root.geometry("560x250")
        self.running = False
        self.mic_stream = None
        self.ref_stream = None
        self.worker_thr = None

        self.btn = ttk.Button(root, text="Start Conversation Mode",
                              command=self.toggle)
        self.btn.pack(pady=16)

        self.prob_lbl = ttk.Label(root, text="Speech Prob (smoothed): 0%")
        self.prob_lbl.pack()

        status = (
            f"MIC={MIC_DEVICE_ID}  REF={REF_DEVICE_ID}  "
            f"I/O={DEVICE_RATE}Hz  VAD={SAMPLE_RATE}Hz  "
            f"Block={FRAME_SIZE} (→ up to {VAD_WINDOW_16K} @16k)  "
            f"Attack={VAD_ATTACK:.2f} Release={VAD_RELEASE:.2f} "
            f"Gain={GAIN_AFTER_AEC}x  Hold={DUCK_HOLD_SILENCE}s"
        )
        self.status_lbl = ttk.Label(root, text=status)
        self.status_lbl.pack(pady=8)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_ui()

    def toggle(self):
        global running_worker

        if not self.running:
            warmup_vad()

            running_worker = True
            self.worker_thr = threading.Thread(target=processing_worker,
                                               daemon=True)
            self.worker_thr.start()

            self.mic_stream = sd.InputStream(
                device=MIC_DEVICE_ID, channels=1, samplerate=DEVICE_RATE,
                callback=mic_callback, blocksize=FRAME_SIZE,
                dtype='float32', latency="high"
            )
            self.ref_stream = sd.InputStream(
                device=REF_DEVICE_ID, channels=1, samplerate=DEVICE_RATE,
                callback=ref_callback, blocksize=FRAME_SIZE,
                dtype='float32', latency="high"
            )
            self.mic_stream.start()
            self.ref_stream.start()
            self.btn.config(text="Stop")
            self.running = True

        else:
            try:
                if self.mic_stream:
                    self.mic_stream.stop(); self.mic_stream.close()
                if self.ref_stream:
                    self.ref_stream.stop(); self.ref_stream.close()
            finally:
                self.mic_stream = None
                self.ref_stream = None
                running_worker = False
                while not audio_q.empty():
                    try:
                        audio_q.get_nowait()
                    except Exception:
                        break
                duck_ctrl.stop()
                self.btn.config(text="Start Conversation Mode")
                self.running = False

    def update_ui(self):
        pct = int(max(0.0, min(1.0, last_vad_prob)) * 100)
        self.prob_lbl.config(text=f"Speech Prob (smoothed): {pct}%")
        self.root.after(100, self.update_ui)

    def on_close(self):
        global running_worker
        if self.running:
            try:
                if self.mic_stream:
                    self.mic_stream.stop(); self.mic_stream.close()
                if self.ref_stream:
                    self.ref_stream.stop(); self.ref_stream.close()
            except Exception:
                pass
        running_worker = False
        try:
            while not audio_q.empty():
                audio_q.get_nowait()
        except Exception:
            pass
        duck_ctrl.stop()
        try:
            if aec_state is not None and aec_lib is not None:
                aec_lib.aec_free(aec_state)
        except Exception:
            pass
        self.root.destroy()

# ---------------------------
# 11) MAIN
# ---------------------------
if __name__ == "__main__":
    try:
        devs = sd.query_devices()
        print("\nAvailable devices:")
        for i, d in enumerate(devs):
            star = "*" if i == sd.default.device[0] or i == sd.default.device[1] else " "
            print(f"{star} {i:2d} {d['name']}, {d['hostapi']} "
                  f"({d['max_input_channels']} in, {d['max_output_channels']} out)")
        print()
    except Exception as e:
        print(f"Device enumeration failed: {e}")

    root = tk.Tk()
    app = DuckingApp(root)
    root.mainloop()
