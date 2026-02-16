import time
import threading
import torch
import numpy as np
import sounddevice as sd
import subprocess
import os
import shutil
import re

# Load Silero VAD model
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)
(get_speech_timestamps, _, _, _, _) = utils

SAMPLE_RATE = 16000
FRAME_SIZE = 512

prev_state = 'SILENCE'

# Volume control setup for Linux using PulseAudio/ALSA
original_volume = None
ducked_volume = None

def get_current_volume():
    """Get current volume percentage (0-100) using pactl"""
    # Prefer pactl, then pamixer, then amixer
    try:
        if shutil.which('pactl'):
            result = subprocess.run(
                ['pactl', 'get-sink-volume', '@DEFAULT_SINK@'],
                capture_output=True,
                text=True
            )
            for part in result.stdout.split():
                if '%' in part:
                    return int(part.rstrip('%'))

        if shutil.which('pamixer'):
            result = subprocess.run(['pamixer', '--get-volume'], capture_output=True, text=True)
            return int(result.stdout.strip())

        if shutil.which('amixer'):
            result = subprocess.run(['amixer', 'sget', 'Master'], capture_output=True, text=True)
            m = re.search(r"(\d+)%", result.stdout)
            if m:
                return int(m.group(1))
    except Exception as e:
        print(f"Error getting volume: {e}")

    return 50  # Default fallback

def set_volume(volume_percent):
    """Set volume percentage (0-100) using pactl"""
    try:
        if shutil.which('pactl'):
            subprocess.run(['pactl', 'set-sink-volume', '@DEFAULT_SINK@', f'{volume_percent}%'], check=True)
            return

        if shutil.which('pamixer'):
            subprocess.run(['pamixer', '--set-volume', str(int(volume_percent))], check=True)
            return

        if shutil.which('amixer'):
            subprocess.run(['amixer', 'sset', 'Master', f'{int(volume_percent)}%'], check=True)
            return

        raise FileNotFoundError('no supported volume control command found')
    except subprocess.CalledProcessError as e:
        print(f"Error setting volume: {e}")
    except FileNotFoundError:
        print("Error: no supported CLI volume control found. Install pactl, pamixer, or amixer.")

duck_duration = 1.5  # seconds to keep volume low
duck_lock = threading.Lock()
duck_timer = None

def restore_volume():
    with duck_lock:
        global duck_timer
        if original_volume is not None:
            set_volume(original_volume)
            print("🔊 Volume restored")
        duck_timer = None

def duck_volume():
    global duck_timer
    with duck_lock:
        # Cancel any existing timer
        if duck_timer and duck_timer.is_alive():
            duck_timer.cancel()

        if ducked_volume is not None:
            set_volume(ducked_volume)
            print("🔉 Volume lowered for 1.5 seconds")

        # Start timer to restore volume after duck_duration
        duck_timer = threading.Timer(duck_duration, restore_volume)
        duck_timer.start()

def audio_callback(indata, frames, time_info, status):
    global prev_state

    if frames != FRAME_SIZE:
        return

    audio_frame = indata[:, 0].copy()
    audio_tensor = torch.from_numpy(audio_frame)

    try:
        speech_probs = model(audio_tensor, SAMPLE_RATE).flatten()
        is_speech = torch.any(speech_probs > 0.5).item()
    except Exception as e:
        print(f"Model error: {e}")
        return

    if is_speech and prev_state != 'SPEAKING':
        print(f"{time_info.inputBufferAdcTime:.2f}s — SPEAKING detected")
        prev_state = 'SPEAKING'
        duck_volume()
    elif not is_speech and prev_state != 'SILENCE':
        print(f"{time_info.inputBufferAdcTime:.2f}s — SILENCE")
        prev_state = 'SILENCE'

if __name__ == "__main__":
    # Initialize volume levels
    original_volume = get_current_volume()
    ducked_volume = max(10, original_volume // 2)  # Lower to 50% of original volume, min 10%
    
    print("Real-time VAD with volume ducking started. Speak into the mic.")
    print("Press Ctrl+C to stop.")
    print(f"Original volume: {original_volume}%")
    print(f"Ducked volume: {ducked_volume}%")

    try:
        with sd.InputStream(callback=audio_callback, channels=1,
                            samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE):
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\nStopped.")
        restore_volume()  # Ensure volume is restored on exit
