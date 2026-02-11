# conversation_mode_aec_v2.py
# -----------------------------------------------------------------------------
# AEC-cleaned voice activity → Smooth endpoint-volume ducking (Windows).
# - MIC: USB PnP Microphone @ 48 kHz (your index: 30)
# - REF: VB-Audio loopback @ 48 kHz (your index: 29)
# - AEC: aec_engine.dll (sample-by-sample, ref=loopback, mic=mic)
# - VAD: Silero @ 16 kHz
# - Duck: target a specific render endpoint (e.g., Speakers (WILLEN))
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

# CAPTURE indices (from sounddevice query)
MIC_ID  = int(os.getenv("AEC_MIC_ID", "30"))   # USB PnP Microphone
LOOP_ID = int(os.getenv("AEC_LOOP_ID", "29"))  # VB-Audio loopback device

# AEC fixed sample rate
SR_AEC = 48000
BLOCK  = 960         # 20 ms @ 48 kHz (reduce to 480 for lower latency)
DTYPE  = 'float32'

# Silero VAD
SR_VAD        = 16000
VAD_FRAME     = 512
VAD_THRESHOLD = 0.30  # lower = more sensitive

# Ducking parameters
DUCK_RATIO         = 0.50
MIN_BASELINE       = 0.10
duck_duration      = 2.0
EPS                = 0.01
USER_DEVIATE_EPS   = 0.03
RAMP_RATIOS        = [0.75, 0.62, DUCK_RATIO]
RAMP_STEP_INTERVAL = 0.06

QUEUE_SIZE = 50

# Endpoint selection (set via environment)
#   AEC_SPEAKER_DEVICE_ID = "{0.0.0.00000000}.{GUID}"   # exact ID (most robust)
#   AEC_SPEAKER_MATCH     = "WILLEN"                    # substring (easy)
#   AEC_SPEAKER_SD_INDEX  = "26"                        # resolve name from sd index (your Willen WASAPI out)
SPEAKER_DEVICE_ID = os.getenv("AEC_SPEAKER_DEVICE_ID", "").strip()
SPEAKER_MATCH     = os.getenv("AEC_SPEAKER_MATCH", "").strip()
SPEAKER_SD_INDEX  = os.getenv("AEC_SPEAKER_SD_INDEX", "").strip()

# Listing / debug flags
LIST_SD_DEVICES        = os.getenv("AEC_LIST_SD_DEVICES", "0").strip() == "1"
LIST_ENDPOINTS         = os.getenv("AEC_LIST_ENDPOINTS", "0").strip() == "1"
LIST_ENDPOINTS_VERBOSE = os.getenv("AEC_LIST_ENDPOINTS_VERBOSE", "0").strip() == "1"
DEBUG_MATCH            = os.getenv("AEC_DEBUG_MATCH", "0").strip() == "1"

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
# Endpoint selection helpers (AudioUtilities-only)
# ---------------------------

def list_render_endpoints(verbose=False):
    """
    List ACTIVE render endpoints via AudioUtilities; show IDs if verbose.
    """
    print("[render endpoints]")
    try:
        devices = AudioUtilities.GetAllDevices()
        for d in devices:
            try:
                name = getattr(d, "FriendlyName", "")
                data_flow = getattr(d, "DataFlow", None)  # 'Render'|'Capture' or 0/1
                state = getattr(d, "State", None)         # 1 == active
                if state not in (1, "Active"):
                    continue
                if str(data_flow).lower() not in ("render", "0"):
                    continue
                if verbose:
                    dev_id = getattr(d, "id", None)
                    print(f"{name} | ID={dev_id}")
                else:
                    print(name)
            except Exception:
                continue
    except Exception as e:
        print(f"[warn] Could not enumerate endpoints: {e}")

def _resolve_match_from_sd_index():
    """If AEC_SPEAKER_SD_INDEX is set, use sd device name (trimmed) for matching."""
    global SPEAKER_MATCH
    if not SPEAKER_SD_INDEX:
        return
    try:
        sd_index = int(SPEAKER_SD_INDEX)
        info = sd.query_devices(sd_index)
        raw_name = info.get("name", "") or ""
        # Keep only the part before the first comma:
        # "Speakers (WILLEN), Windows WASAPI" -> "Speakers (WILLEN)"
        trimmed = raw_name.split(",", 1)[0].strip()
        if trimmed:
            print(f"[info] Using sounddevice index {sd_index} → match substring: '{trimmed}'")
            SPEAKER_MATCH = trimmed
    except Exception as e:
        print(f"[warn] Could not resolve sounddevice index {SPEAKER_SD_INDEX}: {e}")

def _collect_candidates(match_substring):
    """Return list of (score, name, device) for render endpoints containing the substring."""
    match_substring = (match_substring or "").lower()
    devices = AudioUtilities.GetAllDevices()
    candidates = []
    for d in devices:
        try:
            name = getattr(d, "FriendlyName", "") or ""
            data_flow = getattr(d, "DataFlow", None)
            # render endpoints only
            if str(data_flow).lower() not in ("render", "0"):
                continue
            if match_substring in name.lower():
                score = 0
                if name.lower().startswith("speakers"):
                    score += 2
                if name.lower().startswith("headphones"):
                    score += 1
                if any(p in name.lower() for p in ("hands-free", "handset", "aux", "internal aux jack")):
                    score -= 2
                candidates.append((score, name, d))
        except Exception:
            continue
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates

def get_render_endpoint():
    """
    Endpoint selection using AudioUtilities:
      1) Exact Device ID (AEC_SPEAKER_DEVICE_ID)
      2) sounddevice index → name substring match (trimmed)
      3) FriendlyName substring match (try to Activate; prefer 'Speakers'/'Headphones')
         If no hit, try core token inside parentheses, e.g., 'WILLEN'.
      4) Default render endpoint (GetSpeakers)
    """
    # 1) Exact device ID
    if SPEAKER_DEVICE_ID:
        try:
            dev = AudioUtilities.GetDevice(SPEAKER_DEVICE_ID)
            if dev is not None:
                return dev
        except Exception as e:
            print(f"[warn] GetDevice by ID failed: {e}")

    # 2) Resolve name from sd index (optional)
    _resolve_match_from_sd_index()

    # 3) Substring matches (try full then core token)
    def try_match_and_activate(substr):
        cands = _collect_candidates(substr)
        if DEBUG_MATCH:
            print(f"[debug] Candidates for '{substr}': {[n for _, n, _ in cands]}")
        for _, nm, dev in cands:
            try:
                iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                _ = cast(iface, POINTER(IAudioEndpointVolume))  # verify activation
                print(f"[info] Matched and activated: {nm}")
                return dev
            except Exception:
                continue
        return None

    if SPEAKER_MATCH:
        # Try the full string first (e.g., "Speakers (WILLEN)")
        dev = try_match_and_activate(SPEAKER_MATCH)
        if dev:
            return dev
        # If not found, try the core token inside parentheses, e.g., "WILLEN"
        if "(" in SPEAKER_MATCH and ")" in SPEAKER_MATCH:
            core = SPEAKER_MATCH.split("(", 1)[1].split(")", 1)[0].strip()
            if core:
                dev = try_match_and_activate(core)
                if dev:
                    return dev
        # As a last attempt, try the entire raw SD name again but lowercased core token
        dev = try_match_and_activate(SPEAKER_MATCH.split(",", 1)[0])
        if dev:
            return dev
        print(f"[warn] No render endpoint names matched: '{SPEAKER_MATCH}'")

    # 4) Default speakers
    try:
        return AudioUtilities.GetSpeakers()
    except Exception as e:
        print(f"[error] Could not fetch default render endpoint: {e}")
        return None

# ---------------------------
# Initialize volume on chosen endpoint
# ---------------------------

endpoint = None
volume = None
try:
    endpoint = get_render_endpoint()
    if endpoint is None:
        raise RuntimeError("No render endpoint available for ducking.")

    interface = endpoint.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    target_name = getattr(endpoint, "FriendlyName", "Unknown")
    target_id   = getattr(endpoint, "id", None)
    if target_id:
        print(f"[duck target] {target_name} | ID={target_id}")
    else:
        print(f"[duck target] {target_name}")
except Exception as e:
    print(f"[warn] Could not activate EndpointVolume on the selected endpoint ({e}); using default speakers.")
    try:
        fallback = AudioUtilities.GetSpeakers()
        interface = fallback.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        print("[duck target] Default Speakers")
    except Exception as e2:
        raise RuntimeError(f"Failed to open any render endpoint for ducking: {e2}")

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
            # Avoid ducking if system volume already low
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
# Audio queues and callbacks
# ---------------------------

mic_q  = queue.Queue(maxsize=QUEUE_SIZE)
ref_q  = queue.Queue(maxsize=QUEUE_SIZE)
drops_mic = 0
drops_ref = 0

def _safe_enqueue(q: queue.Queue, item, counter_name=None):
    global drops_mic, drops_ref
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            _ = q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            if counter_name == 'mic':
                drops_mic += 1
            elif counter_name == 'ref':
                drops_ref += 1

def mic_callback(indata, frames, time_info, status):
    _safe_enqueue(mic_q, indata.copy(), 'mic')

def loop_callback(indata, frames, time_info, status):
    ref_block = indata[:, 0:1].copy()  # mono ref
    _safe_enqueue(ref_q, ref_block, 'ref')

# ---------------------------
# Processing worker
# ---------------------------

running = True

def processing_loop():
    last_is_speech = False
    vad_resid = np.empty((0,), dtype=np.float32)
    t0 = time.time()

    while running:
        try:
            m_block = mic_q.get(timeout=0.15)
            r_block = ref_q.get(timeout=0.15)
        except queue.Empty:
            continue

        n = min(len(m_block), len(r_block))
        if n <= 0:
            continue

        m = m_block[:n, 0]
        r = r_block[:n, 0]

        # AEC
        clean = np.empty(n, dtype=np.float32)
        for i in range(n):
            clean[i] = aec_lib.aec_process(state_ptr, float(r[i]), float(m[i]))

        # Decimate 48k → 16k (factor 3)
        if n >= 3:
            ds = clean[::3].astype(np.float32)
            vad_resid = np.concatenate((vad_resid, ds), axis=0)

        # VAD @ 16k
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

        # occasional drop stats
        now = time.time()
        if now - t0 > 5.0:
            if drops_mic or drops_ref:
                print(f"[drops] mic={drops_mic}, ref={drops_ref}")
            t0 = now

# ---------------------------
# Main
# ---------------------------

def main():
    global running

    if LIST_SD_DEVICES:
        print("=== sounddevice devices ===")
        print(sd.query_devices())
        print("===========================")
        return

    if LIST_ENDPOINTS or LIST_ENDPOINTS_VERBOSE:
        list_render_endpoints(verbose=LIST_ENDPOINTS_VERBOSE)
        return

    print("=== AEC + Autoduck (Silero VAD on echo-cancelled mic) ===")
    print(f"Mic dev: {MIC_ID}, Loopback dev: {LOOP_ID}, SR: {SR_AEC}, Block: {BLOCK}")

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
        time.sleep(0.2)
        with duck_lock:
            _cancel_ramp_locked()
            _cancel_timer_locked()
        restore_volume()
        if state_ptr:
            aec_lib.aec_free(state_ptr)
        print("Exited cleanly.")

if __name__ == "__main__":
    main()