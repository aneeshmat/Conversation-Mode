import time
import threading
import torch
import numpy as np
import sounddevice as sd

# pycaw imports for Windows volume control
from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

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

# Volume control setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Save the original volume level to restore later
original_volume = volume.GetMasterVolumeLevelScalar()  # 0.0 to 1.0
ducked_volume = original_volume / 2.0  # Lower to 50% of original volume

duck_duration = 1.5  # seconds to keep volume low

duck_lock = threading.Lock()
duck_timer = None

def restore_volume():
    with duck_lock:
        volume.SetMasterVolumeLevelScalar(original_volume, None)
        print("🔊 Volume restored")

def duck_volume():
    global duck_timer
    with duck_lock:
        # Cancel any existing timer
        if duck_timer and duck_timer.is_alive():
            duck_timer.cancel()

        volume.SetMasterVolumeLevelScalar(ducked_volume, None)
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

    speech_probs = model(audio_tensor, SAMPLE_RATE).flatten()
    is_speech = torch.any(speech_probs > 0.5).item()

    if is_speech and prev_state != 'SPEAKING':
        print(f"{time_info.inputBufferAdcTime:.2f}s — SPEAKING detected")
        prev_state = 'SPEAKING'
        duck_volume()
    elif not is_speech and prev_state != 'SILENCE':
        print(f"{time_info.inputBufferAdcTime:.2f}s — SILENCE")
        prev_state = 'SILENCE'

if __name__ == "__main__":
    print("Real-time VAD with volume ducking started. Speak into the mic.")
    print("Press Ctrl+C to stop.")

    try:
        with sd.InputStream(callback=audio_callback, channels=1,
                            samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE):
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\n Stopped.")
        restore_volume()  # Ensure volume is restored on exit
