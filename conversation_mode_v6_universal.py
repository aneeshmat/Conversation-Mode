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
    print("💻 Windows detected.")
else:
    import alsaaudio
    print("🍓 Pi/Linux detected.")

# ---------------------------
# LOAD SILERO VAD (Pi Optimized)
# ---------------------------
torch.set_num_threads(1)
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
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
    def _get_volume(): return float(volume_ctrl.GetMasterVolumeLevelScalar())
    def _set_volume(scalar): volume_ctrl.SetMasterVolumeLevelScalar(max(0.0, min(1.0, float(scalar))), None); return True
else:
    def _get_mixer():
        for name in ['Master', 'PCM', 'Speaker', 'Headphone']:
            try: return alsaaudio.Mixer(name)
            except: continue
        return None
    mixer = _get_mixer()
    def _get_volume(): return mixer.getvolume()[0] / 100.0 if mixer else 0.5
    def _set_volume(scalar):
        if mixer: mixer.setvolume(int(max(0.0, min(1.0, float(scalar))) * 100)); return True
        return False

# ---------------------------
# DUCKING PARAMETERS
# ---------------------------
DUCK_RATIO = 0.50
MIN_BASELINE = 0.10
duck_duration = 2
EPS = 0.01
USER_DEVIATE_EPS = 0.03 # If user moves volume by more than 3%, we recalibrate

duck_lock = threading.Lock()
duck_timer = None
is_ducked = False
pre_duck_volume = None
ducked_volume = None
last_vad_prob = 0.0
audio_stream = None

def restore_volume():
    global is_ducked, pre_duck_volume, ducked_volume
    with duck_lock:
        if not is_ducked: return
        
        current = _get_volume()
        # If user manually changed volume while ducked, don't jump back to old volume
        if ducked_volume is not None and abs(current - ducked_volume) > USER_DEVIATE_EPS:
            print("ℹ️ Manual change detected; staying at current volume.")
        else:
            _set_volume(pre_duck_volume)
            print(f"🔊 Restored to {int(pre_duck_volume*100)}%")
            
        is_ducked = False
        pre_duck_volume = None
        ducked_volume = None

def duck_volume():
    global is_ducked, pre_duck_volume, ducked_volume, duck_timer
    
    with duck_lock:
        current = _get_volume()
        
        if is_ducked:
            # CHECK FOR USER VOLUME CHANGE WHILE DUCKED
            # If the user adjusted the slider, 'current' won't match 'ducked_volume'
            if ducked_volume is not None and abs(current - ducked_volume) > USER_DEVIATE_EPS:
                # Recalculate baseline: If user set it to 30% while ducked, 
                # their intended "normal" volume is likely 60% (since 30 is 50% of 60)
                pre_duck_volume = current / DUCK_RATIO
                ducked_volume = current 
                print(f"🔄 Recalibrated baseline to {int(pre_duck_volume*100)}%")
            
            # Reset timer
            if duck_timer: duck_timer.cancel()
            duck_timer = threading.Timer(duck_duration, restore_volume)
            duck_timer.start()
            return

        # FIRST TIME DUCKING
        if current < MIN_BASELINE: return
        
        pre_duck_volume = current
        target = current * DUCK_RATIO
        _set_volume(target)
        ducked_volume = target
        is_ducked = True
        
        if duck_timer: duck_timer.cancel()
        duck_timer = threading.Timer(duck_duration, restore_volume)
        duck_timer.start()
        print(f"⤵️ Ducked to {int(target*100)}%")

def audio_callback(indata, frames, time_info, status):
    global last_vad_prob
    audio_frame = indata[:, 0].astype(np.float32)
    audio_tensor = torch.from_numpy(audio_frame)
    prob = model(audio_tensor, SAMPLE_RATE).item()
    last_vad_prob = prob
    if prob > 0.40: duck_volume()

# --- GUI CODE (Identical to previous but points to new functions) ---
class DuckApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Ducking")
        self.btn = ttk.Button(root, text="Start", command=self.toggle)
        self.btn.pack(pady=20, padx=20)
        self.prob_bar = ttk.Progressbar(root, length=200)
        self.prob_bar.pack(pady=10)
        self.update_gui()
    def toggle(self):
        global audio_stream
        if not audio_stream:
            audio_stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE)
            audio_stream.start(); self.btn.config(text="Stop")
        else:
            audio_stream.stop(); audio_stream = None; self.btn.config(text="Start")
    def update_gui(self):
        self.prob_bar['value'] = last_vad_prob * 100
        self.root.after(100, self.update_gui)

if __name__ == "__main__":
    root = tk.Tk(); app = DuckApp(root); root.mainloop()