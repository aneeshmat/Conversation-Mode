"""
Conversation Mode - Linux (AEC + VAD + Relative Smooth Ducking + Tk GUI)

- Mic: JLAB TALK MICROPHONE (device 5)
- Ref: PipeWire monitor (device 12)
- I/O at 48 kHz (device-native), resample to 16 kHz for Silero VAD
- AEC runs at the device rate (48 kHz)
- Heavy work is off the audio callback (worker thread)
- Ducking:
    * Ducks to (current volume × DUCK_RATIO) with smooth fades
    * Restores to the pre-duck baseline only if the user didn't change volume while ducked
    * If user changes volume, that becomes the new baseline for the next duck
- Volume control prefers pactl (PipeWire/PulseAudio), falls back to amixer (ALSA)

Dependencies:
  pip install numpy sounddevice packaging
  # Torch VAD (current):
  pip install torch torchvision torchaudio
  # OR lighter fallback:
  pip install onnxruntime

Compile the C core:
  gcc -O3 -fPIC -shared aec_core_vad.c -o aec_vad.so
"""

import os
import sys
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
# 1) SETTINGS (Tailored for your devices)
# ---------------------------
MIC_DEVICE_ID = 5           # JLAB TALK MICROPHONE
REF_DEVICE_ID = 12          # PipeWire monitor (reference)

DEVICE_RATE = 48000         # hardware I/O rate (both mic & ref)
SAMPLE_RATE = 16000         # Silero VAD expects 16 kHz

# 64 ms @ 48k = 3072 frames, which resamples to 1024 @ 16k
FRAME_SIZE = 3072           # device frames per callback (at 48k)
VAD_WINDOW_16K = 1024       # VAD context (64 ms at 16k)
VAD_THRESHOLD = 0.45        # (unused directly; kept for UI only)

# VAD hysteresis + smoothing
VAD_ATTACK = 0.60           # start speech when prob >= 0.60
VAD_RELEASE = 0.35          # end speech when prob <= 0.35
VAD_SMOOTHING = 0.6         # 0..1, higher = more smoothing (slower reaction)

# Signal conditioning
GAIN_AFTER_AEC = 2.0        # multiply cleaned audio by this before VAD
HP_ALPHA = 0.995            # DC-blocking high-pass coefficient

# Relative ducking (with smooth fades)
DUCK_RATIO = 0.5            # duck to 50% of CURRENT volume
DUCK_SECS = 2.5             # how long to stay ducked
FADE_STEPS = 6              # steps for smooth fade up/down
FADE_STEP_MS = 30           # ms between fade steps
VOLUME_RESTORE_TOL = 2      # +/- % tolerance to consider "unchanged" while ducked

# Optional: override via env vars
MIC_DEVICE_ID = int(os.getenv("MIC_DEVICE_ID", MIC_DEVICE_ID))
REF_DEVICE_ID = int(os.getenv("REF_DEVICE_ID", REF_DEVICE_ID))

# Prefer higher latency to avoid overflows on Linux/PipeWire/ALSA
sd.default.latency = "high"   # or tuple: (0.12, 0.12)

# ---------------------------
# 2) LOAD AEC CORE (.so)
# ---------------------------
aec_state = None
aec_lib = None

def load_aec_shared_object():
    global aec_lib, aec_state
    try:
        aec_lib = ctypes.CDLL("./aec_vad.so")
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
# 3) VOLUME CONTROL (pactl preferred, amixer fallback)
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
        percent = max(0, min(150, int(percent)))  # allow >100 if amplification is enabled
        out = subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"],
            capture_output=True, text=True, timeout=1.5
        )
        return out.returncode == 0
    except Exception:
        return False

# --- amixer fallback ---
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
            m = re.search(r'\[(\d+)%\]', out.stdout)
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
    """Return current sink volume percent (0..150)."""
    if _use_pactl():
        v = _pactl_get_volume()
        if v >= 0:
            return v
    return _amixer_get_volume()

def set_current_volume_percent(percent: int) -> bool:
    """Set current sink volume percent, clamped. Returns True on success."""
    if _use_pactl():
        return _pactl_set_volume(percent)
    return _amixer_set_volume(percent)

def smooth_set_volume(from_p: int, to_p: int, steps: int = 6, step_ms: int = 30):
    """Smoothly ramp volume from 'from_p' to 'to_p'."""
    steps = max(1, int(steps))
    if steps == 1 or from_p == to_p:
        set_current_volume_percent(int(round(to_p)))
        return
    for t in np.linspace(from_p, to_p, steps):
        set_current_volume_percent(int(round(t)))
        time.sleep(step_ms / 1000.0)

# ---------------------------
# 4) VAD SETUP (Torch Silero with trust; optional ONNX fallback)
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
            vad_session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
            USE_TORCH = False
            print("✅ Silero VAD (ONNX) loaded")
        except Exception as e2:
            print(f"❌ ONNX fallback failed: {e2}")
            print("VAD unavailable. Speech detection will not work.")

setup_vad()

def warmup_vad():
    """Warm the model once to avoid a compute spike right after streams start."""
    # Torch Silero requires exactly 512 samples @ 16k; use that for warmup.
    dummy = np.zeros(512, dtype=np.float32)
    _ = vad_prob_16k(dummy)

def _torch_vad_prob_16k(audio_16k: np.ndarray) -> float:
    """
    Torch Hub Silero expects exactly 512 samples at 16 kHz (or 256 @ 8 kHz).
    If we have >=1024 samples, evaluate two overlapping 512 windows (last 1024) and take max.
    If we have >=512, evaluate the last 512. Otherwise return 0.0.
    """
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
    """Return speech probability for mono float32 16k signal."""
    if USE_TORCH and model is not None:
        return _torch_vad_prob_16k(audio_16k)
    elif (not USE_TORCH) and vad_session is not None:
        # ONNX model generally accepts variable lengths; use full window.
        inp_name = vad_session.get_inputs()[0].name
        x = audio_16k.astype(np.float32)[None, :]
        out = vad_session.run(None, {inp_name: x})[0]
        return float(out.ravel()[0])
    else:
        return 0.0

# ---------------------------
# 5) RESAMPLER + HIGHPASS (DC blocker)
# ---------------------------
def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """1D linear resampler for short frames. x must be float32 mono."""
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
    """
    Classic DC-blocking IIR high-pass:
      y[n] = x[n] - x[n-1] + alpha * y[n-1]
    alpha close to 1.0 -> lower cutoff. Works well for rumble/DC.
    """
    if x.size == 0:
        return x
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    for n in range(1, x.size):
        y[n] = x[n] - x[n-1] + alpha * y[n-1]
    return y

# ---------------------------
# 6) GLOBALS (Queues, Buffers, VAD state)
# ---------------------------
last_vad_prob = 0.0          # displayed (smoothed) VAD
smoothed_vad = 0.0           # internal smoothed value for hysteresis
speech_active = False        # hysteresis state
is_ducked = False            # ducking in progress

# Reference buffer at DEVICE_RATE (48k), length = FRAME_SIZE
ref_buffer = np.zeros(FRAME_SIZE, dtype=np.float32)

# Worker queue: each item is (mic_frame_48k, ref_frame_48k)
audio_q = queue.Queue(maxsize=16)
running_worker = False

def duck_volume():
    """
    Duck relative to CURRENT volume, with smooth fades:
      - Capture baseline at duck start
      - Fade to ducked = baseline * DUCK_RATIO
      - After DUCK_SECS, if user didn't touch volume (still ~ducked), fade back to baseline
        otherwise respect user's change and do not restore.
    """
    global is_ducked
    is_ducked = True

    baseline = get_current_volume_percent()
    if baseline < 0:
        # Can't read volume; fail gracefully without changing volume
        time.sleep(DUCK_SECS)
        is_ducked = False
        return

    target = int(round(baseline * DUCK_RATIO))
    target = max(0, min(150, target))  # allow >100 if pactl supports amplification

    # Smooth fade down
    smooth_set_volume(baseline, target, steps=FADE_STEPS, step_ms=FADE_STEP_MS)

    # Stay ducked
    time.sleep(DUCK_SECS)

    # If user hasn't moved the volume while ducked (still ~target), smooth fade back
    current = get_current_volume_percent()
    if current >= 0 and abs(current - target) <= VOLUME_RESTORE_TOL:
        smooth_set_volume(current, baseline, steps=FADE_STEPS, step_ms=FADE_STEP_MS)
    # else: user changed volume while ducked; respect it (don't restore)

    is_ducked = False

# ---------------------------
# 7) AUDIO CALLBACKS (lightweight)
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
    mic_frame = indata[:, 0].astype(np.float32).copy()   # 48k, FRAME_SIZE
    ref_frame = ref_buffer.copy()                         # 48k, FRAME_SIZE
    try:
        audio_q.put_nowait((mic_frame, ref_frame))
    except queue.Full:
        # Drop oldest to keep latency bounded
        try:
            _ = audio_q.get_nowait()
            audio_q.put_nowait((mic_frame, ref_frame))
        except Exception:
            pass

# ---------------------------
# 8) PROCESSING WORKER (AEC → highpass/gain → resample → VAD → hysteresis → duck)
# ---------------------------
def processing_worker():
    global last_vad_prob, smoothed_vad, speech_active, is_ducked, running_worker
    while running_worker:
        try:
            mic_dev, ref_dev = audio_q.get(timeout=0.5)
        except queue.Empty:
            continue

        # AEC @ 48k (FRAME_SIZE samples)
        cleaned_dev = np.zeros_like(mic_dev, dtype=np.float32)
        if aec_state is not None and aec_lib is not None:
            aec_lib.aec_process_buffer(aec_state, ref_dev, mic_dev, cleaned_dev, FRAME_SIZE, 0)
        else:
            cleaned_dev[:] = mic_dev

        # High-pass (DC block) at 48k, then resample to 16k
        cleaned_dev = highpass_dc_block(cleaned_dev, alpha=HP_ALPHA)
        cleaned_16k = resample_linear(cleaned_dev, src_sr=DEVICE_RATE, dst_sr=SAMPLE_RATE)

        # Gain boost for VAD robustness
        if GAIN_AFTER_AEC != 1.0:
            cleaned_16k = cleaned_16k * np.float32(GAIN_AFTER_AEC)

        # VAD over the last window(s)
        raw_prob = 0.0
        if cleaned_16k.size >= 512:
            raw_prob = vad_prob_16k(cleaned_16k[-VAD_WINDOW_16K:])

        # Exponential smoothing (limits jitter)
        smoothed_vad = (VAD_SMOOTHING * smoothed_vad) + ((1.0 - VAD_SMOOTHING) * raw_prob)
        last_vad_prob = float(max(0.0, min(1.0, smoothed_vad)))

        # Hysteresis: attack/release to decide "speech_active"
        if not speech_active and last_vad_prob >= VAD_ATTACK:
            speech_active = True
            # Trigger ducking when entering speech
            if not is_ducked:
                threading.Thread(target=duck_volume, daemon=True).start()
        elif speech_active and last_vad_prob <= VAD_RELEASE:
            speech_active = False
        # Otherwise, hold state (reduces chatty toggling)

# ---------------------------
# 9) GUI
# ---------------------------
class DuckingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AEC + VAD (Linux)")
        self.root.geometry("520x240")
        self.running = False
        self.mic_stream = None
        self.ref_stream = None
        self.worker_thr = None

        self.btn = ttk.Button(root, text="Start Conversation Mode", command=self.toggle)
        self.btn.pack(pady=16)

        self.prob_lbl = ttk.Label(root, text="Speech Prob (smoothed): 0%")
        self.prob_lbl.pack()

        status = (
            f"MIC={MIC_DEVICE_ID}  REF={REF_DEVICE_ID}  "
            f"I/O={DEVICE_RATE}Hz  VAD={SAMPLE_RATE}Hz  "
            f"Block={FRAME_SIZE} (→ up to {VAD_WINDOW_16K} @16k)  "
            f"Attack={VAD_ATTACK:.2f} Release={VAD_RELEASE:.2f} Gain={GAIN_AFTER_AEC}x"
        )
        self.status_lbl = ttk.Label(root, text=status)
        self.status_lbl.pack(pady=8)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_ui()

    def toggle(self):
        global running_worker

        if not self.running:
            # Warm up VAD before starting audio to avoid spikes
            warmup_vad()

            # Start processing worker
            running_worker = True
            self.worker_thr = threading.Thread(target=processing_worker, daemon=True)
            self.worker_thr.start()

            # Open both streams at device rate (48k) with float32
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
            # Stop streams
            try:
                if self.mic_stream:
                    self.mic_stream.stop(); self.mic_stream.close()
                if self.ref_stream:
                    self.ref_stream.stop(); self.ref_stream.close()
            finally:
                self.mic_stream = None
                self.ref_stream = None
                # Stop worker
                running_worker = False
                # Drain queue quickly to let thread exit
                while not audio_q.empty():
                    try: audio_q.get_nowait()
                    except Exception: break
                self.btn.config(text="Start Conversation Mode")
                self.running = False

    def update_ui(self):
        pct = int(max(0.0, min(1.0, last_vad_prob)) * 100)
        self.prob_lbl.config(text=f"Speech Prob (smoothed): {pct}%")
        self.root.after(100, self.update_ui)

    def on_close(self):
        # Clean shutdown
        if self.running:
            try:
                if self.mic_stream:
                    self.mic_stream.stop(); self.mic_stream.close()
                if self.ref_stream:
                    self.ref_stream.stop(); self.ref_stream.close()
            except Exception:
                pass
        # Stop worker
        global running_worker
        running_worker = False
        try:
            while not audio_q.empty():
                audio_q.get_nowait()
        except Exception:
            pass
        # Free AEC state
        try:
            if aec_state is not None and aec_lib is not None:
                aec_lib.aec_free(aec_state)
        except Exception:
            pass
        self.root.destroy()

# ---------------------------
# 10) MAIN
# ---------------------------
if __name__ == "__main__":
    # Optional: print device summary once
    try:
        devs = sd.query_devices()
        print("\nAvailable devices:")
        for i, d in enumerate(devs):
            star = "*" if i == sd.default.device[0] or i == sd.default.device[1] else " "
            print(f"{star} {i:2d} {d['name']}, {d['hostapi']} ({d['max_input_channels']} in, {d['max_output_channels']} out)")
        print()
    except Exception as e:
        print(f"Device enumeration failed: {e}")

    root = tk.Tk()
    app = DuckingApp(root)
    root.mainloop()
