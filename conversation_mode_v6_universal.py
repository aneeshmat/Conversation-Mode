import time
import threading
import torch
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import messagebox, ttk
import platform

# ---------------------------
# OS-SPECIFIC SETUP
# ---------------------------
CURRENT_OS = platform.system()

if CURRENT_OS == "Windows":
    from ctypes import POINTER, cast
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    print("💻 Windows detected. Using pycaw for volume.")
else:
    import alsaaudio
    print("🍓 Linux/Pi detected. Using alsaaudio for volume.")

# ---------------------------
# LOAD SILERO VAD (Optimized)
# ---------------------------
# On Pi 3A+, we MUST limit threads to avoid freezing the CPU
torch.set_num_threads(1)
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)
(get_speech_timestamps, _, _, _, _) = utils

SAMPLE_RATE = 16000
FRAME_SIZE = 512

# ---------------------------
# CROSS-PLATFORM VOLUME WRAPPERS
# ---------------------------
if CURRENT_OS == "Windows":
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))

    def _get_volume():
        return float(volume_ctrl.GetMasterVolumeLevelScalar())

    def _set_volume(scalar):
        volume_ctrl.SetMasterVolumeLevelScalar(max(0.0, min(1.0, float(scalar))), None)
        return True
else:
    # PI VOLUME SETUP
    def _get_mixer():
        for name in ['Master', 'PCM', 'Speaker', 'Headphone']:
            try:
                return alsaaudio.Mixer(name)
            except alsaaudio.ALSAAudioError:
                continue
        return None

    mixer = _get_mixer()

    def _get_volume():
        return mixer.getvolume()[0] / 100.0 if mixer else 0.5

    def _set_volume(scalar):
        if mixer:
            mixer.setvolume(int(max(0.0, min(1.0, float(scalar))) * 100))
            return True
        return False

# ---------------------------
# DUCKING LOGIC
# ---------------------------
DUCK_RATIO = 0.50
DUCK_DURATION = 2.0
duck_lock = threading.Lock()
duck_timer = None
is_ducked = False
pre_duck_volume = None
audio_stream = None
last_vad_prob = 0.0

def restore_volume():
    global is_ducked, pre_duck_volume
    with duck_lock:
        if not is_ducked: return
        _set_volume(pre_duck_volume)
        is_ducked = False
        print("🔊 Volume Restored")

def duck_volume():
    global is_ducked, pre_duck_volume, duck_timer
    with duck_lock:
        if not is_ducked:
            pre_duck_volume = _get_volume()
            _set_volume(pre_duck_volume * DUCK_RATIO)
            is_ducked = True
            print("⤵️ Ducking...")
        
        if duck_timer: duck_timer.cancel()
        duck_timer = threading.Timer(DUCK_DURATION, restore_volume)
        duck_timer.start()

def audio_callback(indata, frames, time_info, status):
    global last_vad_prob
    if status: print(status)
    audio_frame = indata[:, 0].astype(np.float32)
    audio_tensor = torch.from_numpy(audio_frame)
    prob = model(audio_tensor, SAMPLE_RATE).item()
    last_vad_prob = prob
    
    if prob > 0.45: # Sensitivity threshold
        duck_volume()

# ---------------------------
# GUI
# ---------------------------
class DuckingApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Universal Ducker ({CURRENT_OS})")
        
        self.status_var = tk.StringVar(value="Status: Ready")
        ttk.Label(root, textvariable=self.status_var).pack(pady=5)
        
        self.btn = ttk.Button(root, text="Start Listening", command=self.toggle)
        self.btn.pack(padx=20, pady=10)
        
        self.prob_bar = ttk.Progressbar(root, length=200, maximum=100)
        self.prob_bar.pack(pady=10)
        
        self.active = False
        self.update_gui()

    def toggle(self):
        global audio_stream
        if not self.active:
            try:
                audio_stream = sd.InputStream(callback=audio_callback, channels=1, 
                                            samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE)
                audio_stream.start()
                self.active = True
                self.btn.config(text="Stop")
                self.status_var.set("Status: Listening...")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            if audio_stream: audio_stream.stop()
            self.active = False
            self.btn.config(text="Start Listening")
            self.status_var.set("Status: Stopped")

    def update_gui(self):
        self.prob_bar['value'] = last_vad_prob * 100
        self.root.after(100, self.update_gui)

if __name__ == "__main__":
    root = tk.Tk()
    app = DuckingApp(root)
    root.mainloop()