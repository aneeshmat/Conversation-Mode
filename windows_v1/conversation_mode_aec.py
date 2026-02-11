# aec_ducking.py
# -----------------------------------------------------------------------------
# AEC-cleaned voice activity → Smooth system-volume ducking (Windows).
#
# - Captures MIC (mono) and WASAPI LOOPBACK (stereo) at 48 kHz
# - Runs your C AEC (aec_engine.dll) sample-by-sample (ref=loopback, mic=mic)
# - Downsamples cleaned signal to 16 kHz and feeds Silero VAD
# - On speech, ducks system volume smoothly using Pycaw (with ramp + hold)
#
# Robustness:
# - Audio callbacks are non-blocking and NEVER raise (drop-oldest strategy)
# - Slightly deeper queues; optional larger block to reduce callback rate
#
# Requirements:
#   pip install numpy sounddevice torch pycaw comtypes
#   (First Silero run will download the model via torch.hub)
#
# Notes:
# - Keep SR_AEC = 48000 because your DLL uses delay_samples = 1920 (~40 ms @ 48k).
# - Adjust MIC_ID and LOOP_ID to your device IDs (WASAPI loopback for speakers).
# -----------------------------------------------------------------------------

import os
import time
import threading
import queue
import ctypes
import numpy as np
import sounddevice as sd
import torch

# Windows volume control (Pycaw)
from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ---------------------------
# Configuration
# ---------------------------

# Device IDs — set to your actual IDs
MIC_ID = int(os.getenv("AEC_MIC_ID", "30"))   # your microphone device id
LOOP_ID = int(os.getenv("AEC_LOOP_ID", "29")) # your WASAPI loopback device id

# AEC runs at 48k to match the DLL's fixed delay of 1920 samples (~40 ms)
SR_AEC = 48000

# Use a slightly larger block (20 ms) to reduce callback overhead
BLOCK = 960  # 20 ms @ 48k. You can lower to 480 (10 ms) for less latency.
DTYPE = 'float32'

# Silero VAD runs at 16 kHz
SR_VAD = 16000
VAD_FRAME = 512
VAD_THRESHOLD = 0.30  # lower = more sensitive

# Ducking parameters (smooth ramps)
DUCK_RATIO = 0.50       # final volume ratio during duck
MIN_BASELINE = 0.10     # don't duck if system volume already below this
duck_duration = 2.0     # seconds to hold after last detected speech
EPS = 0.01
USER_DEVIATE_EPS = 0.03
RAMP_RATIOS = [0.75, 0.62, DUCK_RATIO]
RAMP_STEP_INTERVAL = 0.06  # seconds between ramp steps

# Queue sizes (deeper to survive short spikes)
QUEUE_SIZE = 50

# ---------------------------
# Load AEC DLL
# ---------------------------

AEC_DLL_PATH = os.path.abspath("aec_engine.dll")
if not os.path.exists(AEC_DLL_PATH):
    raise FileNotFoundError(
        f"aec_engine.dll not found at {AEC_DLL_PATH}. "
        "Place the DLL in the same folder as this script."
    )

aec_lib = ctypes.CDLL(AEC_DLL_PATH)
aec_lib.aec_create.restype = ctypes.c_void_p
aec_lib.aec_process.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
aec_lib.aec_process.restype = ctypes.c_float
aec_lib.aec_free.argtypes = [ctypes.c_void_p]

state_ptr = aec_lib.aec_create()
if not state_ptr:
    raise RuntimeError("Failed to create AEC state (aec_create returned NULL)")

# ---------------------------
# Load Silero VAD (torch.hub)
# ---------------------------

print("Loading Silero VAD… (first run may download the model)")
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)
(get_speech_timestamps, _, _, _, _) = utils
print("Silero VAD loaded.")

# ---------------------------
# Pycaw volume control
# ---------------------------

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

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

# ---------------------------
# Ducking state machine
# ---------------------------

duck_lock = threading.Lock()
duck_timer = None
duck_timer_deadline = None
is_ducked = False
pre_duck_volume = None
ducked_volume = None
ramp_thread = None
ramp_cancel_event = threading.Event()
ramp_phase = None  # 'down' | 'up' | None

def _cancel_timer_locked():
    global duck_timer, duck_timer_deadline
    if duck_timer and duck_timer.is_alive():
        duck_timer.cancel()
    duck_timer = None
    duck_timer_deadline = None

def _cancel_ramp_locked():
    global ramp_thread, ramp_cancel_event, ramp_phase
    if ramp_thread and ramp_thread.is_alive():
        ramp_cancel_event.set()
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
    global duck_timer, duck_timer_deadline, is_ducked, pre_duck_volume, ducked_volume, ramp_thread, ramp_phase
    with duck_lock:
        current = _get_volume()
        if current is None:
            print("⚠️ Could not read current volume; skip duck.")
            return

        if current < MIN_BASELINE:
            # Avoid ducking if user already set system volume low
            # print(f"ℹ️ Skipping duck: current volume {int(current*100)}% < {int(MIN_BASELINE*100)}% threshold")
            return

        if is_ducked and (ramp_phase == 'down' or (ducked_volume is not None and current <= ducked_volume + EPS)):
            if duck_timer and duck_timer.is_alive():
                duck_timer.cancel()
            duck_timer = threading.Timer(duck_duration, restore_volume)
            duck_timer.start()
            duck_timer_deadline = time.monotonic() + duck_duration
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
        duck_timer_deadline = time.monotonic() + duck_duration

# ---------------------------
# Audio queues and safe enqueue
# ---------------------------

mic_q = queue.Queue(maxsize=QUEUE_SIZE)
ref_q = queue.Queue(maxsize=QUEUE_SIZE)

# for optional diagnostics
drops_mic = 0
drops_ref = 0

def _safe_enqueue(q: queue.Queue, item, counter_name=None):
    """Non-blocking enqueue. If full, drop oldest then best-effort enqueue."""
    global drops_mic, drops_ref
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            _ = q.get_nowait()  # drop oldest
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            if counter_name == 'mic':
                drops_mic += 1
            elif counter_name == 'ref':
                drops_ref += 1
            # silently drop newest

# ---------------------------
# PortAudio callbacks (NEVER raise)
# ---------------------------

def mic_callback(indata, frames, time_info, status):
    # if status: print(status)  # optional
    _safe_enqueue(mic_q, indata.copy(), 'mic')

def loop_callback(indata, frames, time_info, status):
    # if status: print(status)  # optional
    ref_block = indata[:, 0:1].copy()  # use left channel as mono reference
    _safe_enqueue(ref_q, ref_block, 'ref')

# ---------------------------
# Processing worker
# ---------------------------

running = True

def processing_loop():
    """
    Pull synchronized blocks, run AEC, decimate to 16k, run VAD, trigger ducking.
    """
    last_is_speech = False
    vad_resid = np.empty((0,), dtype=np.float32)

    # For optional throttled diagnostics
    t0 = time.time()

    while running:
        try:
            m_block = mic_q.get(timeout=0.15)  # (N, 1)
            r_block = ref_q.get(timeout=0.15)  # (N, 1)
        except queue.Empty:
            continue

        n = min(len(m_block), len(r_block))
        if n <= 0:
            continue

        m = m_block[:n, 0]
        r = r_block[:n, 0]

        # AEC: sample-by-sample
        clean = np.empty(n, dtype=np.float32)
        for i in range(n):
            clean[i] = aec_lib.aec_process(state_ptr, float(r[i]), float(m[i]))

        # Downsample 48k → 16k (factor 3 decimation). Good enough for VAD.
        if n >= 3:
            ds = clean[::3].astype(np.float32)
            # Efficient buffer extend without frequent realloc: append via list?
            # Here, small concat is OK; blocks are modest.
            vad_resid = np.concatenate((vad_resid, ds), axis=0)

        # Process VAD in 512-sample frames @ 16k
        while len(vad_resid) >= VAD_FRAME:
            frame = vad_resid[:VAD_FRAME]
            vad_resid = vad_resid[VAD_FRAME:]

            with torch.no_grad():
                audio_tensor = torch.from_numpy(frame.copy())
                speech_probs = model(audio_tensor, SR_VAD).flatten()
                max_prob = float(torch.max(speech_probs).item())
                is_speech = max_prob > VAD_THRESHOLD

            if is_speech and not last_is_speech:
                duck_volume()
            last_is_speech = is_speech

        # Optional: print drop stats occasionally (every ~5s)
        now = time.time()
        if now - t0 > 5.0:
            if drops_mic or drops_ref:
                print(f"[drops] mic={drops_mic}, ref={drops_ref}")
            t0 = now

# ---------------------------
# Device helper (optional)
# ---------------------------

def list_devices():
    print(sd.query_devices())

# ---------------------------
# Main
# ---------------------------

def main():
    global running
    print("=== AEC + Autoduck (Silero VAD on echo-cancelled mic) ===")
    print(f"Mic dev: {MIC_ID}, Loopback dev: {LOOP_ID}, SR: {SR_AEC}, Block: {BLOCK}")

    # Open both input streams at 48k for AEC
    try:
        with sd.InputStream(device=MIC_ID, channels=1, samplerate=SR_AEC,
                            dtype=DTYPE, blocksize=BLOCK, callback=mic_callback):
            with sd.InputStream(device=LOOP_ID, channels=2, samplerate=SR_AEC,
                                dtype=DTYPE, blocksize=BLOCK, callback=loop_callback):
                worker = threading.Thread(target=processing_loop, daemon=True)
                worker.start()
                print("Running… Press Ctrl+C to stop.")
                while True:
                    time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping…")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        running = False
        # give worker a moment to finish
        time.sleep(0.2)
        # Cancel any ramps/timers and restore volume if needed
        with duck_lock:
            _cancel_ramp_locked()
            _cancel_timer_locked()
        restore_volume()
        if state_ptr:
            aec_lib.aec_free(state_ptr)
        print("Exited cleanly.")

if __name__ == "__main__":
    # Uncomment to list devices once and set MIC_ID / LOOP_ID:
    # list_devices()
    main()