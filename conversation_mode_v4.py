# smooth ducking
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
DUCK_RATIO = 0.50
MIN_BASELINE = 0.10
duck_duration = 2
EPS = 0.01
USER_DEVIATE_EPS = 0.03

RAMP_RATIOS = [0.75, 0.62, DUCK_RATIO]
RAMP_STEP_INTERVAL = 0.06

duck_lock = threading.Lock()
duck_timer = None

is_ducked = False
pre_duck_volume = None
ducked_volume = None

ramp_thread = None
ramp_cancel_event = threading.Event()
ramp_phase = None  # 'down' | 'up' | None


def _get_volume():
    try:
        return float(volume.GetMasterVolumeLevelScalar())
    except Exception as e:
        print(f"[warn] Get volume failed: {e}")
        return None


def _set_volume(scalar):
    s = max(0.0, min(1.0, float(scalar)))
    try:
        volume.SetMasterVolumeLevelScalar(s, None)
        return True
    except Exception as e:
        print(f"[warn] Set volume failed: {e}")
        return False


def _cancel_timer_locked():
    global duck_timer
    if duck_timer and duck_timer.is_alive():
        duck_timer.cancel()
    duck_timer = None


def _cancel_ramp_locked():
    """
    Cancel any ongoing ramp safely.
    Prevents self-join if called from inside the ramp thread.
    """
    global ramp_thread, ramp_cancel_event, ramp_phase

    if ramp_thread and ramp_thread.is_alive():
        ramp_cancel_event.set()

        # Avoid joining the current thread
        if threading.current_thread() is not ramp_thread:
            ramp_thread.join(timeout=0.25)

    ramp_thread = None
    ramp_cancel_event.clear()
    ramp_phase = None


def _ramp_to_levels(levels, label, on_complete=None):
    last_set = _get_volume()
    if last_set is None:
        last_set = levels[0] if levels else 0.0

    for i, target in enumerate(levels):
        if ramp_cancel_event.is_set():
            return

        current = _get_volume()
        if current is not None and abs(current - last_set) > USER_DEVIATE_EPS:
            print(f"ℹ️ {label}: user adjusted volume (Δ≈{abs(current - last_set):.2f}); stopping ramp")
            return

        ok = _set_volume(target)
        if ok:
            last_set = target
            if i == len(levels) - 1:
                print(f"{label}: reached {int(target*100)}%")
        else:
            print(f"⚠️ {label}: failed to set volume during ramp step")

        if i < len(levels) - 1:
            step_sleep = RAMP_STEP_INTERVAL
            slept = 0.0
            while slept < step_sleep and not ramp_cancel_event.is_set():
                time.sleep(min(0.01, step_sleep - slept))
                slept += 0.01

    if on_complete:
        on_complete()


def restore_volume():
    global is_ducked, pre_duck_volume, ducked_volume, ramp_thread, ramp_phase

    with duck_lock:
        if not is_ducked:
            return

        _cancel_ramp_locked()

        current = _get_volume()
        if current is None:
            if pre_duck_volume is not None:
                _set_volume(pre_duck_volume)
                print("🔊 Volume restored (best-effort)")
            is_ducked = False
            pre_duck_volume = None
            ducked_volume = None
            return

        if ducked_volume is None or abs(current - ducked_volume) > EPS:
            print("ℹ️ Skipped restore (user changed volume during duck)")
            is_ducked = False
            pre_duck_volume = None
            ducked_volume = None
            return

        baseline = pre_duck_volume if pre_duck_volume is not None else current

        ratios = list(RAMP_RATIOS)
        if abs(ratios[-1] - DUCK_RATIO) > 1e-6:
            ratios.append(DUCK_RATIO)

        restore_ratios = list(reversed(ratios[:-1])) + [1.0]
        restore_levels = [max(0.0, min(1.0, baseline * r)) for r in restore_ratios]

        def _on_restore_done():
            # IMPORTANT: do NOT call _cancel_ramp_locked() here
            # because this callback runs inside the ramp thread.
            global pre_duck_volume, ducked_volume, is_ducked, ramp_phase
            with duck_lock:
                is_ducked = False
                ramp_phase = None
                print("🔊 Volume restored smoothly")
                pre_duck_volume = None
                ducked_volume = None

        ramp_phase = 'up'
        label = "⤴️ Restore"
        ramp_thread = threading.Thread(
            target=_ramp_to_levels,
            args=(restore_levels, label, _on_restore_done),
            daemon=True
        )
        ramp_thread.start()


def duck_volume():
    global duck_timer, is_ducked, pre_duck_volume, ducked_volume, ramp_thread, ramp_phase

    with duck_lock:
        current = _get_volume()
        if current is None:
            print("⚠️ Could not read current volume; skip duck.")
            return

        if current < MIN_BASELINE:
            print(f"ℹ️ Skipping duck: current volume {int(current*100)}% < {int(MIN_BASELINE*100)}% threshold")
            return

        if is_ducked and (ramp_phase == 'down' or (ducked_volume is not None and current <= ducked_volume + EPS)):
            if duck_timer and duck_timer.is_alive():
                duck_timer.cancel()
            duck_timer = threading.Timer(duck_duration, restore_volume)
            duck_timer.start()
            print("↪️ Already ducked; extending duck window")
            return

        if is_ducked and ramp_phase == 'up':
            _cancel_ramp_locked()
            baseline = pre_duck_volume if pre_duck_volume is not None else current
        else:
            _cancel_timer_locked()
            _cancel_ramp_locked()
            baseline = current
            pre_duck_volume = current

        ratios = list(RAMP_RATIOS)
        if abs(ratios[-1] - DUCK_RATIO) > 1e-6:
            ratios.append(DUCK_RATIO)
        ratios = [max(0.0, min(1.0, r)) for r in ratios]
        down_levels = [max(0.0, min(1.0, baseline * r)) for r in ratios]

        is_ducked = True
        ramp_phase = 'down'

        def _on_duck_done():
            global ducked_volume, ramp_phase
            final = down_levels[-1] if down_levels else None
            with duck_lock:
                ducked_volume = final
                ramp_phase = None

        label = "⤵️ Duck"
        ramp_thread = threading.Thread(
            target=_ramp_to_levels,
            args=(down_levels, label, _on_duck_done),
            daemon=True
        )
        ramp_thread.start()

        if duck_timer and duck_timer.is_alive():
            duck_timer.cancel()
        duck_timer = threading.Timer(duck_duration, restore_volume)
        duck_timer.start()


def audio_callback(indata, frames, time_info, status):
    global prev_state

    if frames != FRAME_SIZE:
        return

    audio_frame = indata[:, 0].copy()
    audio_tensor = torch.from_numpy(audio_frame)

    speech_probs = model(audio_tensor, SAMPLE_RATE).flatten()
    is_speech = torch.any(speech_probs > 0.5).item() # using a lower threshold for more sensitivity; adjust as needed

    if is_speech and prev_state != 'SPEAKING':
        print(f"{time_info.inputBufferAdcTime:.2f}s — SPEAKING detected")
        prev_state = 'SPEAKING'
        duck_volume()
    elif not is_speech and prev_state == 'SPEAKING':
        print(f"{time_info.inputBufferAdcTime:.2f}s — SILENCE")
        prev_state = 'SILENCE'


if __name__ == "__main__":
    print("Real-time VAD with smooth ducking AND smooth restore started. Speak into the mic.")
    print("Press Ctrl+C to stop.")

    try:
        with sd.InputStream(callback=audio_callback, channels=1,
                            samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE):
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        with duck_lock:
            _cancel_ramp_locked()
            if duck_timer and duck_timer.is_alive():
                duck_timer.cancel()
        restore_volume()