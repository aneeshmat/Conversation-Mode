import time
import threading
import torch
import numpy as np
import sounddevice as sd
import subprocess
import os
from collections import deque
try:
    import tkinter as tk
    GUI_AVAILABLE = True
except Exception:
    GUI_AVAILABLE = False

# Load Silero VAD model
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)
(get_speech_timestamps, _, _, _, _) = utils

SAMPLE_RATE = 16000
FRAME_SIZE = 512

# VAD tuning
VAD_ON_THRESHOLD = float(os.getenv('VAD_ON_THRESHOLD', '0.35'))
VAD_OFF_THRESHOLD = float(os.getenv('VAD_OFF_THRESHOLD', '0.25'))
VAD_EMA_ALPHA = float(os.getenv('VAD_EMA_ALPHA', '0.2'))
VAD_HOLD_FRAMES = int(os.getenv('VAD_HOLD_FRAMES', '6'))  # ~190ms at 512/16k
DEBUG_EVERY_N = int(os.getenv('DEBUG_EVERY_N', '50'))  # print debug every N frames

# AEC / NLMS config
REF_DEVICE = os.getenv('REF_DEVICE', None)  # set to your monitor device name or leave None to disable AEC
FILTER_LEN = 1024  # length of adaptive filter in samples (~64ms for 16kHz)
NLMS_MU = 0.8
NLMS_EPS = 1e-6
USE_GUI = os.getenv('USE_GUI', '1') != '0'

# AEC toggle (GUI controlled)
aec_enabled = True
aec_lock = threading.Lock()

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
vad_prob_ema = None
vad_hold = 0
frame_count = 0

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
    global prev_state, vad_prob_ema, vad_hold, frame_count

    if frames != FRAME_SIZE:
        return

    audio_frame = indata[:, 0].astype(np.float32).copy()
    frame_count += 1

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

    with aec_lock:
        aec_on = aec_enabled

    if use_aec and aec_on:
        try:
            processed = nlms.process(ref_for_processing, audio_frame)
            audio_tensor = torch.from_numpy(processed)
        except Exception as e:
            print(f"AEC processing error: {e}")
            audio_tensor = torch.from_numpy(audio_frame)
    else:
        audio_tensor = torch.from_numpy(audio_frame)

    with torch.no_grad():
        speech_probs = model(audio_tensor, SAMPLE_RATE).flatten()
        frame_prob = float(torch.mean(speech_probs).item())

    if DEBUG_EVERY_N > 0 and (frame_count % DEBUG_EVERY_N == 0):
        rms = float(np.sqrt(np.mean(np.square(audio_frame))))
        print(
            f"VAD debug — rms={rms:.4f} prob={frame_prob:.3f} ema={vad_prob_ema if vad_prob_ema is not None else frame_prob:.3f}"
        )

    if vad_prob_ema is None:
        vad_prob_ema = frame_prob
    else:
        vad_prob_ema = (VAD_EMA_ALPHA * frame_prob) + ((1.0 - VAD_EMA_ALPHA) * vad_prob_ema)

    if vad_prob_ema >= VAD_ON_THRESHOLD:
        is_speech = True
        vad_hold = VAD_HOLD_FRAMES
    elif vad_prob_ema < VAD_OFF_THRESHOLD:
        if vad_hold > 0:
            vad_hold -= 1
            is_speech = True
        else:
            is_speech = False
    else:
        # Between thresholds: keep previous decision, but decay hold
        if vad_hold > 0:
            vad_hold -= 1
            is_speech = True
        else:
            is_speech = (prev_state == 'SPEAKING')

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

    def set_aec_enabled(value):
        global aec_enabled
        with aec_lock:
            aec_enabled = bool(value)

    def toggle_aec():
        with aec_lock:
            current = aec_enabled
        set_aec_enabled(not current)
        print(f"AEC enabled: {not current}")

    def run_audio():
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
            pass
        finally:
            restore_volume()

    audio_thread = threading.Thread(target=run_audio, daemon=True)
    audio_thread.start()

    try:
        has_display = bool(os.getenv('DISPLAY'))
        if USE_GUI and GUI_AVAILABLE and has_display:
            print("GUI enabled — toggle AEC in the window.")
            root = tk.Tk()
            root.title("Conversation Mode")
            root.geometry("280x140")

            aec_var = tk.BooleanVar(value=True)

            title = tk.Label(root, text="AEC / NLMS", font=("Arial", 14))
            title.pack(pady=10)

            toggle = tk.Checkbutton(
                root,
                text="Enable Echo Cancellation",
                variable=aec_var,
                command=lambda: set_aec_enabled(aec_var.get())
            )
            toggle.pack(pady=5)

            hint = tk.Label(root, text="Uncheck to test base VAD", font=("Arial", 9))
            hint.pack(pady=5)

            root.mainloop()
        else:
            print("GUI not available — press Enter to toggle AEC.")
            while True:
                input()
                toggle_aec()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        restore_volume()
