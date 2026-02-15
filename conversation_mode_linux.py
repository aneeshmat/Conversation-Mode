import time
import threading
import torch
import numpy as np
import sounddevice as sd
import subprocess
import os
from collections import deque

# Load Silero VAD model
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)
(get_speech_timestamps, _, _, _, _) = utils

SAMPLE_RATE = 16000
FRAME_SIZE = 512

# AEC / NLMS config
REF_DEVICE = os.getenv('REF_DEVICE', None)  # set to your monitor device name or leave None to disable AEC
FILTER_LEN = 1024  # length of adaptive filter in samples (~64ms for 16kHz)
NLMS_MU = 0.8
NLMS_EPS = 1e-6

# Reference buffer used by NLMS (stores recent speaker output samples)
ref_buffer = deque(maxlen=FILTER_LEN * 8)  # keep several frames worth
ref_lock = threading.Lock()

class NLMSFilter:
    def __init__(self, L=FILTER_LEN, mu=NLMS_MU, eps=NLMS_EPS):
        self.L = L
        self.mu = mu
        self.eps = eps
        self.w = np.zeros(L, dtype=np.float32)

    def process(self, ref_hist, mic_frame):
        """Perform sample-by-sample NLMS using ref_hist (length >= L+N-1) and mic_frame (length N).
        Returns the error signal (mic minus estimated echo) as a numpy array.
        """
        N = len(mic_frame)
        L = self.L
        out = np.zeros(N, dtype=np.float32)

        # ref_hist should be shape (L + N - 1,), where ref_hist[i] corresponds to sample at time i
        for n in range(N):
            x = ref_hist[n:n+L]  # older -> newer
            # reverse so x[0] is newest sample aligning with w[0]
            x_vec = x[::-1]
            y_hat = np.dot(self.w, x_vec)
            e = mic_frame[n] - y_hat
            norm = np.dot(x_vec, x_vec) + self.eps
            self.w += (self.mu / norm) * e * x_vec
            out[n] = e

        return out

# instantiate filter
nlms = NLMSFilter()

prev_state = 'SILENCE'

# Volume control setup for Linux using PulseAudio/ALSA
original_volume = None
ducked_volume = None

def get_current_volume():
    """Get current volume percentage (0-100) using pactl"""
    try:
        result = subprocess.run(
            ['pactl', 'get-sink-volume', '@DEFAULT_SINK@'],
            capture_output=True,
            text=True
        )
        # Parse output like: "Volume: front-left: 65536 / 100% / 0.00 dB"
        for part in result.stdout.split():
            if '%' in part:
                return int(part.rstrip('%'))
    except Exception as e:
        print(f"Error getting volume: {e}")
    return 50  # Default fallback

def set_volume(volume_percent):
    """Set volume percentage (0-100) using pactl"""
    try:
        subprocess.run(
            ['pactl', 'set-sink-volume', '@DEFAULT_SINK@', f'{volume_percent}%'],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error setting volume: {e}")
    except FileNotFoundError:
        print("Error: pactl not found. Please install PulseAudio or PipeWire.")

duck_duration = 1.5  # seconds to keep volume low
duck_lock = threading.Lock()
duck_timer = None

def restore_volume():
    with duck_lock:
        global duck_timer
        set_volume(original_volume)
        print("🔊 Volume restored")
        duck_timer = None

def duck_volume():
    global duck_timer
    with duck_lock:
        # Cancel any existing timer
        if duck_timer and duck_timer.is_alive():
            duck_timer.cancel()

        set_volume(ducked_volume)
        print("🔉 Volume lowered for 1.5 seconds")

        # Start timer to restore volume after duck_duration
        duck_timer = threading.Timer(duck_duration, restore_volume)
        duck_timer.start()

def audio_callback(indata, frames, time_info, status):
    global prev_state

    if frames != FRAME_SIZE:
        return

    audio_frame = indata[:, 0].astype(np.float32).copy()

    # Apply acoustic echo cancellation if reference buffer has enough samples
    use_aec = False
    with ref_lock:
        if len(ref_buffer) >= FILTER_LEN + FRAME_SIZE:
            # build ref_hist of length FILTER_LEN + FRAME_SIZE - 1
            # we take the most recent (FILTER_LEN + FRAME_SIZE - 1) samples
            ref_hist = np.frombuffer(np.array(list(ref_buffer), dtype=np.float32), dtype=np.float32)
            # ensure we have at least L + N - 1 samples
            if len(ref_hist) >= FILTER_LEN + FRAME_SIZE - 1:
                # align so that the last sample in ref_hist corresponds to the last sample of mic_frame
                start = len(ref_hist) - (FILTER_LEN + FRAME_SIZE - 1)
                ref_segment = ref_hist[start: start + FILTER_LEN + FRAME_SIZE - 1]
                # For NLMS processing we need L + N -1; pad if necessary
                if len(ref_segment) >= FILTER_LEN + FRAME_SIZE - 1:
                    use_aec = True
                    ref_for_processing = ref_segment

    if use_aec:
        try:
            processed = nlms.process(ref_for_processing, audio_frame)
            audio_tensor = torch.from_numpy(processed)
        except Exception as e:
            print(f"AEC processing error: {e}")
            audio_tensor = torch.from_numpy(audio_frame)
    else:
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
    # Initialize volume levels
    original_volume = get_current_volume()
    ducked_volume = max(10, original_volume // 2)  # Lower to 50% of original volume, min 10%
    
    print("Real-time VAD with volume ducking started. Speak into the mic.")
    print("Press Ctrl+C to stop.")
    print(f"Original volume: {original_volume}%")
    print(f"Ducked volume: {ducked_volume}%")

    # If a reference device is provided, start a background stream to capture speaker output
    def ref_callback(indata, frames, time_info, status):
        # store reference (speaker) samples into ref_buffer
        samples = indata[:, 0].astype(np.float32).copy()
        with ref_lock:
            ref_buffer.extend(samples.tolist())

    try:
        if REF_DEVICE:
            print(f"Starting reference capture on device: {REF_DEVICE}")
            ref_stream = sd.InputStream(callback=ref_callback, channels=1,
                                         samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE,
                                         device=REF_DEVICE)
            mic_stream = sd.InputStream(callback=audio_callback, channels=1,
                                        samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE)
            with ref_stream, mic_stream:
                print("Real-time VAD with AEC and volume ducking started. Speak into the mic.")
                while True:
                    sd.sleep(100)
        else:
            print("Reference device not set — running without AEC.")
            with sd.InputStream(callback=audio_callback, channels=1,
                                samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE):
                print("Real-time VAD with volume ducking started. Speak into the mic.")
                while True:
                    sd.sleep(100)
    except KeyboardInterrupt:
        print("\nStopped.")
        restore_volume()  # Ensure volume is restored on exit
