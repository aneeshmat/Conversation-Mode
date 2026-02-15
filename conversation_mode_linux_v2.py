import time
import threading
import torch
import numpy as np
import sounddevice as sd
import subprocess
import re
import os

from collections import dequeue

try:
  import tkinter as tk
  GUI_AVAILABLE = True
except Exception:
  GUI_AVAILABLE = False

model, utils = torch.hub.load(repo_or_dir = 'snakers4/silero-vad', model = 'silero_vad', force_reload = False)
(get_speech_timestamps, _, _, _, _) = utils

SAMPLE_RATE = 16000
FRAME_SIZE = 512

import sounddevice as sd

def auto_select_input_device(required_channels=1, required_samplerate=16000):
    devices = sd.query_devices()
    candidates = []

    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] >= required_channels:
            try:
                sd.check_input_settings(
                    device=idx,
                    channels=required_channels,
                    samplerate=required_samplerate
                )
                candidates.append((idx, dev))
            except Exception:
                pass

    if not candidates:
        return None

    # Prefer non-monitor devices first
    candidates.sort(key=lambda x: ("monitor" in x[1]["name"].lower()))
    return candidates[0][0]

MIC_ID = auto_select_input_device()

def auto_select_reference_device(required_samplerate=48000):
    devices = sd.query_devices()
    candidates = []

    keywords = ["monitor", "loopback", "loop", "mix", "echo"]

    for idx, dev in enumerate(devices):
        name = dev["name"].lower()

        # Must be an input-capable device
        if dev["max_input_channels"] == 0:
            continue

        # Must match loopback/monitor keywords
        if any(k in name for k in keywords):
            try:
                sd.check_input_settings(
                    device=idx,
                    samplerate=required_samplerate,
                    channels=1
                )
                candidates.append((idx, dev))
            except Exception:
                pass

    if not candidates:
        return None

    # Prefer PulseAudio/PipeWire monitors over ALSA loopbacks
    candidates.sort(key=lambda x: ("monitor" not in x[1]["name"].lower()))
    return candidates[0][0]
REF_ID = auto_select_reference_device()

def get_system_volume():
    # 1. Try PipeWire (wpctl)
    try:
        out = subprocess.check_output(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"], text=True)
        # Example: "Volume: 0.32 [0.00, 1.00]"
        m = re.search(r'Volume:\s*([0-9.]+)', out)
        if m:
            return float(m.group(1)) * 100
    except Exception:
        pass

    # 2. Try PulseAudio (pactl)
    try:
        out = subprocess.check_output(["pactl", "get-sink-volume", "@DEFAULT_SINK@"], text=True)
        # Example: "Volume: front-left: 65536 / 100% / 0.00 dB"
        m = re.search(r'(\d+)%', out)
        if m:
            return int(m.group(1))
    except Exception:
        pass

    # 3. Try ALSA (amixer)
    try:
        out = subprocess.check_output(["amixer", "get", "Master"], text=True)
        # Example: "[75%]"
        m = re.search(r'\[(\d+)%\]', out)
        if m:
            return int(m.group(1))
    except Exception:
        pass

    return None
