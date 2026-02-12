#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conversation Mode - Linux (AEC + VAD + Volume Ducking + Tk GUI)

- Mic: JLAB TALK MICROPHONE (device 5)
- Ref: PipeWire monitor (device 12)
- I/O at 48 kHz (device-native), resample to 16 kHz for Silero VAD
- AEC runs at the device rate (48 kHz)
- Volume ducking via ALSA (tries Master, PCM, Speaker, Headphone)

Dependencies (typical):
  pip install numpy sounddevice torch torchvision torchaudio packaging
  # OR switch to ONNX (lighter): pip install onnxruntime

Compile the C core:
  gcc -O3 -fPIC -shared aec_core_vad.c -o aec_vad.so
"""

import os
import sys
import time
import threading
import subprocess
import ctypes

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

# NOTE: At 48k, 32 ms = 1536 frames, which converts to exactly 512 samples @16k.
# You can use FRAME_SIZE=1536 to get one VAD decision per callback with no accumulation.
FRAME_SIZE = 1536            # current device frames per callback (at 48k)
VAD_WINDOW_16K = 512        # 32 ms window for Silero VAD
VAD_THRESHOLD = 0.45        # speech probability threshold

DUCK_LOW = 20               # percent
DUCK_HIGH = 80              # percent
DUCK_SECS = 2.5             # seconds

# Optional: override via environment variables if needed
MIC_DEVICE_ID = int(os.getenv("MIC_DEVICE_ID", MIC_DEVICE_ID))
REF_DEVICE_ID = int(os.getenv("REF_DEVICE_ID", REF_DEVICE_ID))

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
    _try_set_volume("Master", percent)

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
# 6) GLOBALS FOR CALLBACKS
# ---------------------------
last_vad_prob = 0.0
is_ducked = False

# ref buffer at DEVICE_RATE (48k)
ref_buffer = np.zeros(FRAME_SIZE, dtype=np.float32)

# Ring buffer at 16k for VAD; we only call VAD when full (>=512 samples)
vad_ring = np.zeros(VAD_WINDOW_16K, dtype=np.float32)
vad_filled = 0  # number of valid samples currently in vad_ring (0..512)

def duck_volume():
    global is_ducked
    is_ducked = True
    set_linux_volume(DUCK_LOW)
    time.sleep(DUCK_SECS)
    set_linux_volume(DUCK_HIGH)
    is_ducked = False

# ---------------------------
# 7) AUDIO CALLBACKS
# ---------------------------
def ref_callback(indata, frames, time_info, status):
    if status:
        print("Ref status:", status)
    global ref_buffer
    # keep reference at device rate (48k)
    ref = indata[:, 0].astype(np.float32)
    # ensure the buffer has FRAME_SIZE samples
    if len(ref) >= FRAME_SIZE:
        ref_buffer[:] = ref[:FRAME_SIZE]
    else:
        ref_buffer[:len(ref)] = ref
        ref_buffer[len(ref):] = 0.0

def mic_callback(indata, frames, time_info, status):
    if status:
        print("Mic status:", status)
    global last_vad_prob, is_ducked, vad_ring, vad_filled

    # Mic at device rate (48k)
    mic_raw_dev = indata[:, 0].astype(np.float32)
    cleaned_dev = np.zeros(FRAME_SIZE, dtype=np.float32)

    # AEC operates at DEVICE_RATE (48k)
    if aec_state is not None and aec_lib is not None:
        aec_lib.aec_process_buffer(
            aec_state, ref_buffer,
            mic_raw_dev[:FRAME_SIZE],
            cleaned_dev, FRAME_SIZE, 0
        )
    else:
        cleaned_dev[:len(mic_raw_dev[:FRAME_SIZE])] = mic_raw_dev[:FRAME_SIZE]

    # Resample 48k -> 16k for VAD
    cleaned_16k = resample_linear(cleaned_dev, src_sr=DEVICE_RATE, dst_sr=SAMPLE_RATE)

    # Append to the 16k ring buffer
    n = len(cleaned_16k)
    if n >= VAD_WINDOW_16K:
        # take last VAD_WINDOW_16K samples directly
        vad_ring[:] = cleaned_16k[-VAD_WINDOW_16K:]
        vad_filled = VAD_WINDOW_16K
    else:
        if vad_filled < VAD_WINDOW_16K:
            take = min(n, VAD_WINDOW_16K - vad_filled)
            vad_ring[vad_filled:vad_filled + take] = cleaned_16k[:take]
            vad_filled += take
        else:
            # roll left and append at end
            shift = n
            if shift > 0:
                vad_ring = np.roll(vad_ring, -shift)
                vad_ring[-shift:] = cleaned_16k

    # Only call VAD when we have at least 512 samples (avoids "too short" error)
    prob = last_vad_prob
    if vad_filled >= VAD_WINDOW_16K:
        try:
            prob = vad_prob_16k(vad_ring)
        except Exception as e:
            # Guard against model raising inside audio thread
            # Print once in a while if needed, but don't spam
            # print(f"VAD error: {e}")
            prob = last_vad_prob

    last_vad_prob = float(prob)

    # Trigger ducking if speech detected
    if last_vad_prob > VAD_THRESHOLD and not is_ducked:
        threading.Thread(target=duck_volume, daemon=True).start()

# ---------------------------
# 8) GUI
# ---------------------------
class DuckingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AEC + VAD (Linux)")
        self.root.geometry("360x180")
        self.running = False
        self.mic_stream = None
        self.ref_stream = None

        self.btn = ttk.Button(root, text="Start Conversation Mode", command=self.toggle)
        self.btn.pack(pady=16)

        self.prob_lbl = ttk.Label(root, text="Speech Prob: 0%")
        self.prob_lbl.pack()

        self.status_lbl = ttk.Label(root, text=f"MIC={MIC_DEVICE_ID} REF={REF_DEVICE_ID} I/O={DEVICE_RATE}Hz VAD={SAMPLE_RATE}Hz")
        self.status_lbl.pack(pady=8)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_ui()

    def toggle(self):
        if not self.running:
            # Try to open both streams at device rate (48k)
            self.mic_stream = sd.InputStream(
                device=MIC_DEVICE_ID, channels=1, samplerate=DEVICE_RATE,
                callback=mic_callback, blocksize=FRAME_SIZE
            )
            self.ref_stream = sd.InputStream(
                device=REF_DEVICE_ID, channels=1, samplerate=DEVICE_RATE,
                callback=ref_callback, blocksize=FRAME_SIZE
            )
            self.mic_stream.start()
            self.ref_stream.start()
            self.btn.config(text="Stop")
            self.running = True
        else:
            try:
                if self.mic_stream:
                    self.mic_stream.stop()
                    self.mic_stream.close()
                if self.ref_stream:
                    self.ref_stream.stop()
                    self.ref_stream.close()
            finally:
                self.mic_stream = None
                self.ref_stream = None
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
                    self.mic_stream.stop()
                    self.mic_stream.close()
                if self.ref_stream:
                    self.ref_stream.stop()
                    self.ref_stream.close()
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
# 9) MAIN
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
