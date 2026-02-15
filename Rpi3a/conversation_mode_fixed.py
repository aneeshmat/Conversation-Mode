import os
import sys
import threading
import queue
import ctypes
import subprocess
import shutil
import urllib.request
import zipfile
import io
import re
import time

import numpy as np
import sounddevice as sd

import tkinter as tk
from tkinter import ttk

# -----------------------------
# Global flags and state
# -----------------------------
running = False
audio_thread = None
audio_q = queue.Queue()

vad_model = None          # Torch model
vad_session = None        # ONNX session
onnx_input_names = []     # ONNX input names

aec_lib = None

# -----------------------------
# AEC shared object loading
# -----------------------------
def load_aec_shared_object():
    global aec_lib
    so_name = "libaec.so"
    so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), so_name)

    if not os.path.isfile(so_path):
        print(f"⚠️ AEC shared object not found at {so_path}")
        return

    try:
        aec_lib = ctypes.CDLL(so_path)
        print("✅ AEC Shared Object Loaded")
    except Exception as e:
        print(f"❌ Failed to load AEC shared object: {e}")
        aec_lib = None


# -----------------------------
# Silero VAD download helpers
# -----------------------------
def download_silero_vad_onnx(model_path="silero_vad.onnx"):
    if os.path.isfile(model_path):
        return model_path

    print("Downloading Silero VAD ONNX model...")
    url = "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
    urllib.request.urlretrieve(url, model_path)
    print("✅ Silero VAD ONNX downloaded")
    return model_path


def download_silero_vad_torch():
    # Uses torch hub to download Silero VAD
    import torch
    torch.hub.set_dir(os.path.expanduser("~/.cache/torch/hub"))
    print("Downloading Silero VAD Torch model via torch.hub...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False
    )
    print("✅ Silero VAD Torch model loaded")
    return model, utils


# -----------------------------
# VAD initialization
# -----------------------------
def init_vad():
    global vad_model, vad_session, onnx_input_names

    # Try Torch first
    try:
        import torch
        try:
            import torchaudio  # Silero often expects this
        except ImportError:
            print("⚠️ torchaudio not found, Torch VAD may fail for resampling")

        try:
            vad_model, utils = download_silero_vad_torch()
            vad_model.eval()
            print("✅ Silero VAD (Torch) initialized")
            return
        except Exception as e:
            print(f"⚠️ Torch/Silero VAD load failed: {e}")
            vad_model = None
    except ImportError:
        print("⚠️ torch not installed, skipping Torch VAD")

    # Fallback to ONNX
    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ ONNX fallback failed: onnxruntime not installed")
        vad_session = None
        return

    try:
        model_path = download_silero_vad_onnx()
        vad_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        onnx_input_names = [i.name for i in vad_session.get_inputs()]
        print("✅ Silero VAD (ONNX) loaded")
    except Exception as e:
        print(f"❌ ONNX fallback failed: {e}")
        vad_session = None


# -----------------------------
# Universal VAD probability
# -----------------------------
def vad_prob_16k(x):
    """
    Universal ONNX/Torch VAD wrapper.
    Works with:
      - Torch Silero VAD
      - ONNX 2-input model (input, state)
      - ONNX 3-input model (input, state, sr)
    """
    global vad_model, vad_session, onnx_input_names

    # Torch path
    if vad_model is not None:
        import torch
        with torch.no_grad():
            audio_t = torch.from_numpy(x).float()
            # Silero VAD expects mono 16k
            prob = vad_model(audio_t, 16000).item()
            return float(prob)

    # ONNX path
    if vad_session is not None:
        feed = {}

        for name in onnx_input_names:
            if name == "input":
                feed[name] = x.astype("float32")
            elif name == "state":
                # Silero state is [2, 1, 128] for standard models
                feed[name] = np.zeros((2, 1, 128), dtype=np.float32)
            elif name == "sr":
                feed[name] = np.array([16000], dtype=np.int64)

        out = vad_session.run(None, feed)[0]
        return float(out.squeeze())

    # No VAD available
    return 0.0


def warmup_vad():
    """Run a dummy inference to warm up the VAD model."""
    dummy = np.zeros(16000, dtype=np.float32)
    try:
        _ = vad_prob_16k(dummy)
        print("🔧 VAD warmup complete")
    except Exception as e:
        print(f"⚠️ VAD warmup failed: {e}")


# -----------------------------
# Audio processing
# -----------------------------
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
CHANNELS = 1


def list_devices():
    print("\nAvailable devices:")
    devices = sd.query_devices()
    default = sd.default.device
    for idx, dev in enumerate(devices):
        mark = "*" if idx == default[0] or idx == default[1] else " "
        print(f"{mark} {idx:2d} {dev['name']}, {dev['hostapi']} ({dev['max_input_channels']} in, {dev['max_output_channels']} out)")
    print("")


def audio_callback(indata, outdata, frames, time_info, status):
    if status:
        print(f"Audio callback status: {status}", file=sys.stderr)

    # Mono
    mono = indata[:, 0].copy()

    # AEC would go here if you wire it up via aec_lib
    # For now, just pass through
    outdata[:, 0] = mono

    # Push to queue for VAD / logging
    try:
        audio_q.put_nowait(mono.copy())
    except queue.Full:
        pass


def audio_loop():
    global running

    with sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback
    ):
        print("🎙️ Audio stream started")
        while running:
            try:
                block = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # VAD on each block
            prob = vad_prob_16k(block)
            # You can add smoothing / thresholds here
            # For now, just print occasionally
            if prob > 0.5:
                print(f"Speech detected, VAD prob={prob:.2f}")
        print("🛑 Audio loop exiting")


# -----------------------------
# GUI
# -----------------------------
def toggle():
    global running, audio_thread

    if not running:
        running = True
        warmup_vad()
        audio_thread = threading.Thread(target=audio_loop, daemon=True)
        audio_thread.start()
        btn_toggle.config(text="Stop")
    else:
        running = False
        btn_toggle.config(text="Start")


def on_close():
    global running
    running = False
    root.destroy()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    load_aec_shared_object()
    init_vad()
    list_devices()

    # Tkinter GUI
    root = tk.Tk()
    root.title("Conversation Mode with AEC + VAD")

    mainframe = ttk.Frame(root, padding="10")
    mainframe.grid(row=0, column=0, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    lbl = ttk.Label(mainframe, text="Conversation Mode (AEC + VAD)")
    lbl.grid(row=0, column=0, pady=(0, 10))

    btn_toggle = ttk.Button(mainframe, text="Start", command=toggle)
    btn_toggle.grid(row=1, column=0, pady=5)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
