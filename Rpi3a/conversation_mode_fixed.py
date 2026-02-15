#!/usr/bin/env python3
import os
import sys
import threading
import queue
import ctypes
import urllib.request

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

vad_model = None          # Torch Silero VAD model
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
# Silero VAD (Torch) helpers
# -----------------------------
def download_silero_vad_torch():
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


def init_vad():
    global vad_model
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        import packaging  # noqa: F401
    except ImportError as e:
        print(f"❌ Torch VAD dependencies missing: {e}")
        print("   Install with: pip install torch torchaudio packaging")
        vad_model = None
        return

    try:
        vad_model, utils = download_silero_vad_torch()
        vad_model.eval()
        print("✅ Silero VAD (Torch) initialized")
    except Exception as e:
        print(f"❌ Torch/Silero VAD load failed: {e}")
        vad_model = None


# -----------------------------
# VAD probability (Torch only)
# -----------------------------
def vad_prob_16k(x: np.ndarray) -> float:
    """
    Torch Silero VAD wrapper.
    Expects mono 16k audio.
    """
    global vad_model

    if vad_model is None:
        return 0.0

    import torch

    # Ensure mono 1D
    if x.ndim > 1:
        x = x[:, 0]
    audio_t = torch.from_numpy(x).float()

    with torch.no_grad():
        prob = vad_model(audio_t, 16000).item()
    return float(prob)


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
        print(
            f"{mark} {idx:2d} {dev['name']} "
            f"({dev['max_input_channels']} in, {dev['max_output_channels']} out)"
        )
    print("")


def audio_callback(indata, outdata, frames, time_info, status):
    if status:
        print(f"Audio callback status: {status}", file=sys.stderr)

    # Mono
    mono = indata[:, 0].copy()

    # AEC hook (if you wire it later via aec_lib)
    # For now, pass-through
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

            prob = vad_prob_16k(block)
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
    root.title("Conversation Mode with AEC + Torch VAD")

    mainframe = ttk.Frame(root, padding="10")
    mainframe.grid(row=0, column=0, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    lbl = ttk.Label(mainframe, text="Conversation Mode (AEC + Torch VAD)")
    lbl.grid(row=0, column=0, pady=(0, 10))

    btn_toggle = ttk.Button(mainframe, text="Start", command=toggle)
    btn_toggle.grid(row=1, column=0, pady=5)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
