import time
import threading
import torch
import numpy as np
import sounddevice as sd
from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ---------------------------
# NLMS AEC Algorithm Class
# ---------------------------
class NLMSEchoCanceller:
    def __init__(self, filter_len=1024, mu=0.1, eps=1e-6):
        self.filter_len = filter_len
        self.mu = mu  # Step size (adaptation rate)
        self.eps = eps # Stability constant to avoid division by zero
        self.w = np.zeros(filter_len) # Adaptive filter weights
        self.x_history = np.zeros(filter_len) # Reference signal buffer

    def process(self, mic_signal, ref_signal):
        """Processes a block of audio. Signals must be same length."""
        out = np.zeros_like(mic_signal)
        for i in range(len(mic_signal)):
            # Update reference history
            self.x_history = np.roll(self.x_history, 1)
            self.x_history[0] = ref_signal[i]

            # Predict echo: y = w.T * x
            y_hat = np.dot(self.w, self.x_history)

            # Error signal (Clean audio): e = d - y_hat
            e = mic_signal[i] - y_hat

            # NLMS Weight Update: w = w + mu * (e * x) / (||x||^2 + eps)
            norm_x = np.dot(self.x_history, self.x_history) + self.eps
            self.w += self.mu * e * self.x_history / norm_x
            
            out[i] = e
        return out

# ---------------------------
# Setup & Config
# ---------------------------
SAMPLE_RATE = 16000
FRAME_SIZE = 512

# AEC Instance
aec = NLMSEchoCanceller(filter_len=1024, mu=0.1)

# Buffer to store loopback audio (Speaker output)
# We need this to match the mic input timing
ref_buffer = np.zeros(FRAME_SIZE)

# VAD Model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)

# Volume Control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

original_volume = volume.GetMasterVolumeLevelScalar()
ducked_volume = original_volume / 2.0
duck_duration = 1.5
duck_lock = threading.Lock()
duck_timer = None
prev_state = 'SILENCE'

# ---------------------------
# Logic Functions
# ---------------------------
def restore_volume():
    with duck_lock:
        volume.SetMasterVolumeLevelScalar(original_volume, None)
        print("🔊 Volume restored")

def duck_volume():
    global duck_timer
    with duck_lock:
        if duck_timer and duck_timer.is_alive():
            duck_timer.cancel()
        volume.SetMasterVolumeLevelScalar(ducked_volume, None)
        print("🔉 Volume lowered")
        duck_timer = threading.Timer(duck_duration, restore_volume)
        duck_timer.start()

# ---------------------------
# Callbacks
# ---------------------------
def loopback_callback(indata, frames, time_info, status):
    """Captures speaker output to use as AEC reference."""
    global ref_buffer
    ref_buffer = indata[:, 0].copy()

def mic_callback(indata, frames, time_info, status):
    """Captures mic, runs AEC, then runs VAD."""
    global prev_state, ref_buffer

    mic_audio = indata[:, 0].copy()
    
    # 1. RUN NLMS ECHO CANCELLATION
    # Subtract ref_buffer (Speaker) from mic_audio (Mic)
    clean_audio = aec.process(mic_audio, ref_buffer)

    # 2. RUN VAD ON CLEAN AUDIO
    audio_tensor = torch.from_numpy(clean_audio.astype(np.float32))
    speech_probs = model(audio_tensor, SAMPLE_RATE).flatten()
    is_speech = torch.any(speech_probs > 0.5).item()

    if is_speech and prev_state != 'SPEAKING':
        print("🎙️ Speech detected (Echo Filtered)")
        prev_state = 'SPEAKING'
        duck_volume()
    elif not is_speech and prev_state != 'SILENCE':
        prev_state = 'SILENCE'

# ---------------------------
# Main Loop
# ---------------------------
if __name__ == "__main__":
    # Note: On Windows, to use loopback, you must select your output device 
    # as an INPUT device using the WASAPI host.
    print("AEC + VAD Active. Adjusting for speaker noise...")

    try:
        # Stream 1: Speaker Loopback (Reference)
        # Note: 'device' may need to be your specific WASAPI Loopback ID
        with sd.InputStream(callback=loopback_callback, channels=1, 
                            samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE):
            
            # Stream 2: Microphone Input
            with sd.InputStream(callback=mic_callback, channels=1, 
                                samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE):
                
                while True:
                    sd.sleep(100)
                    
    except KeyboardInterrupt:
        print("\nStopped.")
        restore_volume()