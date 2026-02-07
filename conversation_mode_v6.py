# with aec
import time
import threading
import torch
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

# pycaw imports for Windows volume control
from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ---------------------------
# Load Silero VAD model
# ---------------------------
# NOTE: First run may take a moment as PyTorch Hub fetches the model.
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
duck_duration = 2  # seconds to remain ducked after last speech
EPS = 0.01
USER_DEVIATE_EPS = 0.03

RAMP_RATIOS = [0.75, 0.62, DUCK_RATIO]
RAMP_STEP_INTERVAL = 0.06

duck_lock = threading.Lock()
duck_timer = None
duck_timer_deadline = None  # for metrics: time.monotonic() when duck window ends

is_ducked = False
pre_duck_volume = None
ducked_volume = None

ramp_thread = None
ramp_cancel_event = threading.Event()
ramp_phase = None  # 'down' | 'up' | None

# Handle the live stream for start/stop from the GUI
audio_stream = None

# Live VAD metrics
last_vad_prob = 0.0     # max speech prob in the last processed frame
last_is_speech = False  # True/False for last frame


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
    global duck_timer, duck_timer_deadline
    if duck_timer and duck_timer.is_alive():
        duck_timer.cancel()
    duck_timer = None
    duck_timer_deadline = None


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
    global duck_timer, duck_timer_deadline, is_ducked, pre_duck_volume, ducked_volume, ramp_thread, ramp_phase

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
            duck_timer_deadline = time.monotonic() + duck_duration
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
        duck_timer_deadline = time.monotonic() + duck_duration


def audio_callback(indata, frames, time_info, status):
    global prev_state, last_vad_prob, last_is_speech

    if frames != FRAME_SIZE:
        return

    # Ensure float32 for the model
    audio_frame = indata[:, 0].astype(np.float32).copy()
    audio_tensor = torch.from_numpy(audio_frame)

    speech_probs = model(audio_tensor, SAMPLE_RATE).flatten()
    max_prob = float(torch.max(speech_probs).item())
    last_vad_prob = max_prob
    is_speech = max_prob > 0.3  # lower threshold = more sensitive
    last_is_speech = bool(is_speech)

    if is_speech and prev_state != 'SPEAKING':
        # print(f"{time_info.inputBufferAdcTime:.2f}s — SPEAKING detected")
        prev_state = 'SPEAKING'
        duck_volume()
    elif not is_speech and prev_state == 'SPEAKING':
        # print(f"{time_info.inputBufferAdcTime:.2f}s — SILENCE")
        prev_state = 'SILENCE'


# ---------------------------
# GUI wrapper: start/stop safely + metrics
# ---------------------------

def start_listening():
    """Create and start the InputStream if not already running."""
    global audio_stream
    if audio_stream is not None:
        return
    try:
        audio_stream = sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE
        )
        audio_stream.start()
        print("🎙️ Listening started.")
    except Exception as e:
        audio_stream = None
        print(f"Failed to start audio stream: {e}")
        raise


def stop_listening_and_cleanup():
    """Stop the stream, cancel timers/ramps, and restore volume."""
    global audio_stream, prev_state
    try:
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()
            audio_stream = None
            print("🛑 Listening stopped.")
    except Exception as e:
        print(f"Failed to stop/close audio stream: {e}")
    finally:
        with duck_lock:
            _cancel_ramp_locked()
            _cancel_timer_locked()
        # Ensure volume returns to pre-duck level if we were ducked
        restore_volume()
        prev_state = 'SILENCE'


class DuckingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smooth Ducking (Silero VAD)")
        self.enabled = False

        # ---------- Controls ----------
        controls = ttk.Frame(root)
        controls.pack(padx=14, pady=(14, 8), fill="x")

        self.btn = ttk.Button(controls, text="Enable", command=self.toggle, width=18)
        self.btn.pack(side="left")

        self.status_var = tk.StringVar(value="Status: Disabled")
        self.status_lbl = ttk.Label(controls, textvariable=self.status_var)
        self.status_lbl.pack(side="left", padx=12)

        # ---------- Metrics ----------
        metrics = ttk.LabelFrame(root, text="Live Metrics")
        metrics.pack(padx=14, pady=(0, 14), fill="x")

        # System volume
        self.vol_var = tk.StringVar(value="System Volume: N/A")
        self.vol_lbl = ttk.Label(metrics, textvariable=self.vol_var)
        self.vol_lbl.grid(row=0, column=0, sticky="w", padx=8, pady=4)

        self.vol_bar = ttk.Progressbar(metrics, orient="horizontal", length=220, mode="determinate", maximum=100)
        self.vol_bar.grid(row=0, column=1, sticky="w", padx=8, pady=4)

        # Speaking + VAD prob
        self.speaking_var = tk.StringVar(value="Speaking: No (VAD max: 0%)")
        self.speaking_lbl = ttk.Label(metrics, textvariable=self.speaking_var)
        self.speaking_lbl.grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=4)

        # Ducking state / phase
        self.duck_state_var = tk.StringVar(value="Ducking: No | Phase: -")
        self.duck_state_lbl = ttk.Label(metrics, textvariable=self.duck_state_var)
        self.duck_state_lbl.grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=4)

        # Duck timer remaining
        self.duck_timer_var = tk.StringVar(value="Duck time remaining: -")
        self.duck_timer_lbl = ttk.Label(metrics, textvariable=self.duck_timer_var)
        self.duck_timer_lbl.grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=4)

        # Baseline / target duck level
        self.levels_var = tk.StringVar(value="Baseline: - | Duck target: -")
        self.levels_lbl = ttk.Label(metrics, textvariable=self.levels_var)
        self.levels_lbl.grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=4)

        # Info line
        self.info_var = tk.StringVar(value=f"Samplerate: {SAMPLE_RATE} | Frame size: {FRAME_SIZE}")
        self.info_lbl = ttk.Label(root, textvariable=self.info_var)
        self.info_lbl.pack(padx=14, pady=(0, 10), anchor="w")

        # Handle close (X) properly
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start polling metrics
        self.poll_interval_ms = 200
        self.root.after(self.poll_interval_ms, self.poll_metrics)

    def toggle(self):
        if self.enabled:
            self.disable()
        else:
            self.enable()

    def enable(self):
        try:
            start_listening()
            self.enabled = True
            self.btn.configure(text="Disable")
            self.status_var.set("Status: Listening…")
        except Exception as e:
            messagebox.showerror("Start failed", str(e))

    def disable(self):
        stop_listening_and_cleanup()
        self.enabled = False
        self.btn.configure(text="Enable")
        self.status_var.set("Status: Disabled")

    def on_close(self):
        if self.enabled:
            self.disable()
        self.root.destroy()

    def poll_metrics(self):
        """Update the GUI with live metrics (volume, VAD, duck state, etc.)."""
        # Volume
        vol = _get_volume()
        if vol is None:
            self.vol_var.set("System Volume: N/A")
            self.vol_bar['value'] = 0
        else:
            pct = int(round(vol * 100))
            self.vol_var.set(f"System Volume: {pct}%")
            self.vol_bar['value'] = pct

        # VAD / speaking
        vad_prob = max(0.0, min(1.0, float(last_vad_prob)))
        vad_pct = int(round(vad_prob * 100))
        speak_txt = "Yes" if last_is_speech else "No"
        self.speaking_var.set(f"Speaking: {speak_txt} (VAD max: {vad_pct}%)")

        # Ducking state, phase, timer remaining, baseline/target
        with duck_lock:
            ducking_txt = "Yes" if is_ducked else "No"
            phase_txt = ramp_phase if ramp_phase is not None else "-"
            # Time remaining
            if duck_timer is not None and duck_timer.is_alive() and duck_timer_deadline is not None:
                remaining = max(0.0, duck_timer_deadline - time.monotonic())
                self.duck_timer_var.set(f"Duck time remaining: {remaining:.2f} s")
            else:
                self.duck_timer_var.set("Duck time remaining: -")

            # Baseline and target
            baseline = pre_duck_volume
            target = (baseline * DUCK_RATIO) if (baseline is not None) else None

        self.duck_state_var.set(f"Ducking: {ducking_txt} | Phase: {phase_txt}")
        if baseline is None:
            self.levels_var.set("Baseline: - | Duck target: -")
        else:
            b_pct = int(round(baseline * 100))
            t_pct = int(round(max(0.0, min(1.0, target)) * 100)) if target is not None else "-"
            self.levels_var.set(f"Baseline: {b_pct}% | Duck target: {t_pct}%")

        # Keep polling
        self.root.after(self.poll_interval_ms, self.poll_metrics)


if __name__ == "__main__":
    print("GUI mode — click Enable to start, Disable to stop.")
    root = tk.Tk()

    # Optional: make default ttk theme look nicer on Windows
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass

    app = DuckingApp(root)
    root.mainloop()