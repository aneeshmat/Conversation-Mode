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

def get_loopback_device_id():
    try:
        # 1. Run 'aplay -l' to get the hardware list
        output = subprocess.check_output(["aplay", "-l"], text=True)
        
        # 2. Use regex to find the line: "card 2: Loopback [Loopback], device 0: ..."
        # This captures the card number (\d+)
        match = re.search(r"card (\d+): Loopback", output)
        
        if match:
            card_no = match.group(1)
            search_string = f"hw:{card_no}"
            
            # 3. Match the ALSA hardware ID to the sounddevice index
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                # We check for the 'hw:X' pattern in the device name/info
                # and ensure it's an input-capable device
                if search_string in dev['name'] and dev['max_input_channels'] > 0:
                    return i
                    
            # Fallback: if 'hw:X' isn't in name, look for 'Loopback' in sounddevice list
            for i, dev in enumerate(devices):
                if "loopback" in dev['name'].lower() and dev['max_input_channels'] > 0:
                    return i
                    
        print("Loopback card not found in aplay -l. Is 'snd-aloop' loaded?")
        return None
        
    except Exception as e:
        print(f"Error identifying loopback: {e}")
        return None

MIC_ID = auto_select_input_device()

REF_ID = get_loopback_device_id()

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

def audio_loop(mic_id, ref_id, samplerate=16000, framesize=512):
    with sd.InputStream(device=mic_id,
                        channels=1,
                        samplerate=samplerate,
                        blocksize=framesize) as mic_stream, \
         sd.InputStream(device=ref_id,
                        channels=1,
                        samplerate=samplerate,
                        blocksize=framesize) as ref_stream:

        while True:
            mic_frame, _ = mic_stream.read(framesize)
            ref_frame, _ = ref_stream.read(framesize)

            process_audio(mic_frame, ref_frame)

def process_audio(mic_frame, ref_frame):
    """
    mic_frame: numpy array of shape (N, 1) from microphone
    ref_frame: numpy array of shape (N, 1) from loopback/monitor device
    """
    # For now, just a placeholder
    pass
