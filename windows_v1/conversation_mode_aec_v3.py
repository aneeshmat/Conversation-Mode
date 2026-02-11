import time
import threading
import torch
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import messagebox, ttk
import ctypes

# Windows Volume Control
from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL, CoInitialize
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ---------------------------
# 1. SETTINGS
# ---------------------------
MIC_DEVICE_ID = 4   
REF_DEVICE_ID = 26  
SAMPLE_RATE = 16000
FRAME_SIZE = 512
DUCK_RATIO = 0.50

# ---------------------------
# 2. LOAD MODELS & DLL
# ---------------------------
model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)

try:
    aec_lib = ctypes.CDLL("./aec_vad.dll")
    aec_lib.aec_process_buffer.argtypes = [
        ctypes.c_void_p, 
        np.ctypeslib.ndpointer(dtype=np.float32), 
        np.ctypeslib.ndpointer(dtype=np.float32), 
        np.ctypeslib.ndpointer(dtype=np.float32), 
        ctypes.c_int, ctypes.c_int
    ]
    aec_lib.aec_create.restype = ctypes.c_void_p
    aec_state = aec_lib.aec_create()
    print("✅ AEC DLL Loaded Successfully")
except Exception as e:
    print(f"⚠️ AEC DLL Failed: {e}")
    aec_state = None

# ---------------------------
# 3. VOLUME & GLOBAL STATE
# ---------------------------
CoInitialize()
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

last_vad_prob = 0.0
is_ducked = False
ref_buffer = np.zeros(FRAME_SIZE, dtype=np.float32)

# ---------------------------
# 4. AUDIO CALLBACKS (Flexible Channels)
# ---------------------------
def ref_callback(indata, frames, time_info, status):
    global ref_buffer
    # indata[:, 0] works even if the device is stereo or multi-channel
    ref_buffer[:] = indata[:, 0].astype(np.float32)

def mic_callback(indata, frames, time_info, status):
    global last_vad_prob, is_ducked
    
    # Slice first channel to handle stereo/mono gracefully
    mic_raw = indata[:, 0].astype(np.float32)
    cleaned = np.zeros(FRAME_SIZE, dtype=np.float32)

    if aec_state:
        aec_lib.aec_process_buffer(aec_state, ref_buffer, mic_raw, cleaned, FRAME_SIZE, 0)
    else:
        cleaned = mic_raw

    audio_tensor = torch.from_numpy(cleaned)
    prob = float(model(audio_tensor, SAMPLE_RATE).item())
    last_vad_prob = prob

    # Basic ducking trigger
    if prob > 0.45 and not is_ducked:
        threading.Thread(target=duck_volume, daemon=True).start()

# ---------------------------
# 5. DUCKING LOGIC
# ---------------------------
def duck_volume():
    global is_ducked
    try:
        is_ducked = True
        current = volume.GetMasterVolumeLevelScalar()
        # Ducking to 50%
        volume.SetMasterVolumeLevelScalar(current * DUCK_RATIO, None)
        time.sleep(2.5) 
        volume.SetMasterVolumeLevelScalar(current, None)
        is_ducked = False
    except:
        is_ducked = False

# ---------------------------
# 6. GUI CLASS
# ---------------------------
class DuckingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AEC + VAD (Pi 3A Optimized)")
        self.root.geometry("350x200")
        self.running = False
        self.mic_stream = None
        self.ref_stream = None

        self.btn = ttk.Button(root, text="Enable Conversation Mode", command=self.toggle)
        self.btn.pack(pady=20)

        self.prob_lbl = ttk.Label(root, text="Speech Prob: 0%")
        self.prob_lbl.pack()
        
        self.update_ui()

    def toggle(self):
        if not self.running:
            try:
                # Set channels=None to let PortAudio pick the device's default
                self.mic_stream = sd.InputStream(device=MIC_DEVICE_ID, channels=None, samplerate=SAMPLE_RATE, callback=mic_callback, blocksize=FRAME_SIZE)
                self.ref_stream = sd.InputStream(device=REF_DEVICE_ID, channels=None, samplerate=SAMPLE_RATE, callback=ref_callback, blocksize=FRAME_SIZE)
                self.mic_stream.start()
                self.ref_stream.start()
                self.btn.config(text="Disable")
                self.running = True
            except Exception as e:
                messagebox.showerror("Stream Error", f"Could not open devices: {e}")
        else:
            if self.mic_stream: self.mic_stream.stop(); self.mic_stream.close()
            if self.ref_stream: self.ref_stream.stop(); self.ref_stream.close()
            self.btn.config(text="Enable")
            self.running = False

    def update_ui(self):
        self.prob_lbl.config(text=f"Speech Prob: {int(last_vad_prob * 100)}%")
        self.root.after(100, self.update_ui)

if __name__ == "__main__":
    root = tk.Tk()
    app = DuckingApp(root)
    root.mainloop()