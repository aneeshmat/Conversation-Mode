#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conversation Mode - Linux (AEC + VAD + Volume Ducking + Tk GUI)

- Mic: JLAB TALK MICROPHONE (device 5)
- Ref: PipeWire monitor (device 12)
- I/O at 48 kHz (device-native), resample to 16 kHz for Silero VAD
- AEC runs at the device rate (48 kHz)
- Heavy work moved off the audio callback to a worker thread
- Volume ducking via ALSA (auto-picks Master/PCM/Speaker/Headphone)

Dependencies:
  pip install numpy sounddevice packaging
  # For Torch VAD (current setup):
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

# Use 1536 frames at 48k (= 32 ms), which resamples exactly to 512 @ 16k.
FRAME_SIZE = 1536
VAD_WINDOW_16K = 512        # 32 ms window for Silero VAD
VAD_THRESHOLD = 0.45        # speech probability threshold

DUCK_LOW = 20               # percent
DUCK_HIGH = 80              # percent
DUCK_SECS = 2.5             # seconds

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
# 3) ALSA VOLUME CONTROL (auto-pick a mixer control)
# ---------------------------
_already_picked_control = None

def _try_set_volume(control: str, percent: int) -> bool:
    try:
        proc = subprocess.run(
            ["amixer", "sset", control, f"{percent}%"],
            capture_output=True,
            text=True
        )
        return proc.returncode == 0 and "Unable to find simple control" not in proc.stderr
    except Exception:
        return False

def set_linux_volume(percent: int):
    """Try a list of common ALSA simple controls and cache the first one that works."""
    global _already_picked_control
    percent = max(0, min(100, int(percent)))
    if _already_picked_control:
        _try_set_volume(_already_picked_control, percent)
        return
    for name in ["Master", "PCM", "Speaker", "Headphone"]:
        if _try_set_volume(name, percent):
            _already_picked_control = name
            return
    _try_set_volume("Master", percent)  # best-effort fallback

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
        # Trust repo to silence future warnings; uses cache after first download
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
    dummy = np.zeros(VAD_WINDOW_16K, dtype=np.float32)
    _ = vad_prob_16k(dummy)

def vad_prob_16k(audio_16k: np.ndarray) -> float:
    """Return speech probability for mono float32 16k signal. Requires >=512 samples."""
    if audio_16k.size < VAD_WINDOW_16K:
        return 0.0
    if USE_TORCH and model is not None:
        import torch
        with torch.no_grad():
            t = torch.from_numpy(audio_16k.astype(np.float32))
            return float(model(t, SAMPLE_RATE).item())
    elif (not USE_TORCH) and vad_session is not None:
        inp_name = vad_session.get_inputs()[0].name
        x = audio_16k.astype(np.float32)[None, :]
        out = vad_session.run(None, {inp_name: x})[0]
        return float(out.ravel()[0])
    else:
        return 0.0

# ---------------------------
# 5) LIGHTWEIGHT RESAMPLER (48k → 16k)
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

# ---------------------------
# 6) GLOBALS (Queues, Buffers)
# ---------------------------
last_vad_prob = 0.0
is_ducked = False

# Reference buffer at DEVICE_RATE (48k), length = FRAME_SIZE
ref_buffer = np.zeros(FRAME_SIZE, dtype=np.float32)

# Worker queue: each item is (mic_frame_48k, ref_frame_48k)
audio_q = queue.Queue(maxsize=16)
running_worker = False

def duck_volume():
    global is_ducked
    is_ducked = True
    set_linux_volume(DUCK_LOW)
    time.sleep(DUCK_SECS)
    set_linux_volume(DUCK_HIGH)
    is_ducked = False

# ---------------------------
# 7) AUDIO CALLBACKS (lightweight)
# ---------------------------
def ref_callback(indata, frames, time_info, status):
    if status:
        print("Ref status:", status)
    global ref_buffer
    # keep reference at device rate (48k)
    ref = indata[:, 0].astype(np.float32)
    if len(ref) >= FRAME_SIZE:
        ref_buffer[:] = ref[:FRAME_SIZE]
    else:
        # shouldn't happen with fixed blocksize, but keep safe
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
        # Drop oldest by pulling one, then push (keeps latency bounded)
        try:
            _ = audio_q.get_nowait()
            audio_q.put_nowait((mic_frame, ref_frame))
        except Exception:
            pass

# ---------------------------
# 8) PROCESSING WORKER (AEC → resample → VAD → duck)
# ---------------------------
def processing_worker():
    global last_vad_prob, is_ducked, running_worker
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

        # Resample 48k → 16k (with FRAME_SIZE=1536 this is exactly 512 samples)
        cleaned_16k = resample_linear(cleaned_dev, src_sr=DEVICE_RATE, dst_sr=SAMPLE_RATE)

        # VAD (expect >= 512 samples)
        prob = last_vad_prob
        try:
            if cleaned_16k.size >= VAD_WINDOW_16K:
                prob = vad_prob_16k(cleaned_16k[-VAD_WINDOW_16K:])
        except Exception:
            pass

        last_vad_prob = float(prob)

        # Trigger ducking if speech detected
        if last_vad_prob > VAD_THRESHOLD and not is_ducked:
            threading.Thread(target=duck_volume, daemon=True).start()

# ---------------------------
# 9) GUI
# ---------------------------
class DuckingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AEC + VAD (Linux)")
        self.root.geometry("380x200")
        self.running = False
        self.mic_stream = None
        self.ref_stream = None
        self.worker_thr = None

        self.btn = ttk.Button(root, text="Start Conversation Mode", command=self.toggle)
        self.btn.pack(pady=16)

        self.prob_lbl = ttk.Label(root, text="Speech Prob: 0%")
        self.prob_lbl.pack()

        self.status_lbl = ttk.Label(
            root,
            text=f"MIC={MIC_DEVICE_ID}  REF={REF_DEVICE_ID}  I/O={DEVICE_RATE}Hz  VAD={SAMPLE_RATE}Hz  Block={FRAME_SIZE}"
        )
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
        self.prob_lbl.config(text=f"Speech Prob: {pct}%")
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
