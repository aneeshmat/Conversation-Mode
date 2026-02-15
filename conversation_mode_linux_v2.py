import time
import threading
import torch
import numpy as np
import sounddevice as sd
import subprocess
import re
import os

from collections import deque

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

def auto_select_input_device():
    devices = sd.query_devices()
    
    # Strategy 1: Look for the system default first
    try:
        default_in = sd.query_devices(kind='input')
        return default_in['index']
    except:
        pass

    # Strategy 2: Search for specific hardware (Surface Pro mic)
    for i, dev in enumerate(devices):
        name = dev['name'].lower()
        if "mic" in name or "input" in name:
            if dev['max_input_channels'] > 0:
                return i
    return None

def auto_select_reference_device():
    devices = sd.query_devices()
    # On Linux/Pipewire, look for "Monitor" of the output
    for i, dev in enumerate(devices):
        name = dev['name'].lower()
        if "monitor" in name and dev['max_input_channels'] > 0:
            return i
    return None

MIC_ID = auto_select_input_device()

REF_ID = auto_select_reference_device()

print(MIC_ID)
print(REF_ID)

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
