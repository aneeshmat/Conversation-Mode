# minimum threshold
import time
import threading
import torch
import numpy as np
import sounddevice as sd

# pycaw imports for Windows volume control
from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ---------------------------
# Load Silero VAD model
# ---------------------------
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)
(get_speech_timestamps, _, _, _, _) = utils

SAMPLE_RATE = 16000
FRAME_SIZE = 512

prev_state = 'SILENCE'

# ---------------------------
# Volume control setup
# ---------------------------
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# ---------------------------
# Ducking parameters
# ---------------------------
DUCK_RATIO = 0.5          # Duck to 50% of current volume
duck_duration = 1.5       # seconds to keep volume low after last speech
EPS = 0.01                # tolerance for float comparisons [0..1]
MIN_BASELINE = 0.10       # If current volume < 10%, skip ducking entirely

duck_lock = threading.Lock()
duck_timer = None

# State for duck/restore
is_ducked = False
pre_duck_volume = None    # snapshot before duck
ducked_volume = None      # level we set when ducking


def _get_volume():
    """Get current master volume scalar [0..1]."""
    try:
        return float(volume.GetMasterVolumeLevelScalar())
    except Exception as e:
        print(f"[warn] Get volume failed: {e}")
        return None


def _set_volume(scalar):
    """Clamp and set master volume scalar [0..1]."""
    s = max(0.0, min(1.0, float(scalar)))
    try:
        volume.SetMasterVolumeLevelScalar(s, None)
        return True
    except Exception as e:
        print(f"[warn] Set volume failed: {e}")
        return False


def _cancel_timer_locked():
    """Cancel existing duck timer; caller must hold duck_lock."""
    global duck_timer
    if duck_timer and duck_timer.is_alive():
        duck_timer.cancel()
    duck_timer = None


def restore_volume():
    """
    Restore politely:
    - Only restore if the system is still approximately at the ducked level we set.
    - If the user changed the slider during ducking, do not override their choice.
    """
    global is_ducked, pre_duck_volume, ducked_volume
    with duck_lock:
        if not is_ducked:
            return

        current = _get_volume()
        if current is None:
            # Can't read current; best effort restore to pre-duck if we have it
            if pre_duck_volume is not None:
                _set_volume(pre_duck_volume)
                print("🔊 Volume restored (best-effort)")
            is_ducked = False
            pre_duck_volume = None
            ducked_volume = None
            return

        # Only restore if we still "own" the current level (≈ ducked level)
        if ducked_volume is not None and abs(current - ducked_volume) <= EPS:
            if pre_duck_volume is not None:
                _set_volume(pre_duck_volume)
                print("🔊 Volume restored")
        else:
            # User moved the volume; respect their choice
            print("ℹ️ Skipped restore (user changed volume during duck)")

        is_ducked = False
        pre_duck_volume = None
        ducked_volume = None


def duck_volume():
    """
    Duck relative to the *current* master volume.
    - If volume is below MIN_BASELINE, skip ducking.
    - If already ducked, just extend the timer (polite behavior).
    """
    global duck_timer, is_ducked, pre_duck_volume, ducked_volume

    with duck_lock:
        current = _get_volume()
        if current is None:
            print("⚠️ Could not read current volume; skip duck.")
            return

        # Skip ducking when system volume is already very low
        if current < MIN_BASELINE:
            print(f"ℹ️ Skipping duck: current volume {int(current*100)}% < {int(MIN_BASELINE*100)}% threshold")
            return

        # We intend to duck; manage/extend the timer window
        if duck_timer and duck_timer.is_alive():
            duck_timer.cancel()

        if not is_ducked:
            # First time ducking (or after restoration)
            pre_duck_volume = current
            target = max(0.0, min(1.0, current * DUCK_RATIO))

            # Avoid redundant set if already at or below target
            if current - target > EPS:
                ok = _set_volume(target)
                if ok:
                    ducked_volume = target
                    is_ducked = True
                    print(f"🔉 Volume lowered to {int(target*100)}% (from {int(pre_duck_volume*100)}%) for {duck_duration:.1f}s")
                else:
                    print("⚠️ Failed to set ducked volume")
            else:
                ducked_volume = current
                is_ducked = True
                print(f"🔉 Volume already low (≈{int(current*100)}%), extending duck window")
        else:
            # Already ducked — keep steady to avoid oscillations.
            # If you prefer aggressive re-ducking when the user raises during duck,
            # compute a new target and set it here.
            pass

        # Start/update timer to restore after silence
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
    elif not is_speech and prev_state == 'SPEAKING':
        print(f"{time_info.inputBufferAdcTime:.2f}s — SILENCE")
        prev_state = 'SILENCE'
        # Do nothing here; the timer will restore if no new speech
    # else: already SILENCE -> nothing to do


if __name__ == "__main__":
    print("Real-time VAD with dynamic relative volume ducking (+ min-threshold) started. Speak into the mic.")
    print("Press Ctrl+C to stop.")

    try:
        with sd.InputStream(callback=audio_callback, channels=1,
                            samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE):
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        # Ensure timer is cleaned up and volume restored if we are still ducked
        with duck_lock:
            if duck_timer and duck_timer.is_alive():
                duck_timer.cancel()
        restore_volume()
