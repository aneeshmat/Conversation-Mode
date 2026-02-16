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
    try:
        # Priority 1: Get the system's actual default input
        return sd.default.device[0]
    except Exception:
        # Priority 2: Manual search for anything labeled as a Mic or Input
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                name = dev['name'].lower()
                if any(k in name for k in ["mic", "input", "capture"]):
                    return i
    return None

def auto_select_reference_device():
    # 1. Ask PipeWire which sink is active
    try:
        out = subprocess.check_output(["wpctl", "status"], text=True)
        # Find the active sink (marked with "*")
        m = re.search(r'\*\s+(\d+)\.\s+([^\n]+)', out)
        if m:
            sink_name = m.group(2).strip()
            monitor_name = sink_name + ".monitor"

            # 2. Match this monitor to a sounddevice input
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if monitor_name.lower() in dev["name"].lower():
                    return i
    except Exception:
        pass

    # Fallback: keyword search
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            name = dev["name"].lower()
            if "monitor" in name:
                return i

    return None



MIC_ID = auto_select_input_device()

REF_ID = auto_select_reference_device()

print(MIC_ID) #22
print(REF_ID) #6

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

def process_audio(mic_frame, ref_frame):
    # mic_frame: numpy array (FRAME_SIZE, 1)
    # ref_frame: numpy array (FRAME_SIZE, 1)
    print("Mic RMS:", float(np.sqrt(np.mean(mic_frame**2))))
    print("Ref RMS:", float(np.sqrt(np.mean(ref_frame**2))))
    # Later: AEC, VAD, subtraction, etc.

def audio_loop():
    # device=None → use PipeWire default source (contains mic + monitor)
    with sd.InputStream(device=None,
                        channels=2,
                        samplerate=SAMPLE_RATE,
                        blocksize=FRAME_SIZE,
                        dtype='float32') as stream:

        while True:
            frame, _ = stream.read(FRAME_SIZE)

            # Split channels
            mic_frame = frame[:, 0]   # channel 0 = microphone
            ref_frame = frame[:, 1]   # channel 1 = system audio monitor

            print("Mic RMS:", float(np.sqrt(np.mean(mic_frame**2))))
            print("Ref RMS:", float(np.sqrt(np.mean(ref_frame**2))))
print(np.allclose(frame[:,0], frame[:,1]))
print(frame.shape)


audio_loop()
