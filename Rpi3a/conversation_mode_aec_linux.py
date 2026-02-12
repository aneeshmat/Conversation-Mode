import time
import threading
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import messagebox, ttk
import ctypes
import subprocess

# ---------------------------
# 1. SETTINGS
# ---------------------------
# Use 'sd.query_devices()' in terminal to find these IDs on your Pi/Zorin
MIC_DEVICE_ID = 5  
REF_DEVICE_ID = 12  
SAMPLE_RATE = 16000
FRAME_SIZE = 512
DUCK_RATIO = 0.3  # Reduced to 30% volume

# ---------------------------
# 2. LOAD C CORE (.so)
# ---------------------------
try:
    # Look for .so instead of .dll
    aec_lib = ctypes.CDLL("./aec_vad.so")
    aec_lib.aec_process_buffer.argtypes = [
        ctypes.c_void_p, 
        np.ctypeslib.ndpointer(dtype=np.float32), 
        np.ctypeslib.ndpointer(dtype=np.float32), 
        np.ctypeslib.ndpointer(dtype=np.float32), 
        ctypes.c_int, ctypes.c_int
    ]
    aec_lib.aec_create.restype = ctypes.c_void_p
    aec_state = aec_lib.aec_create()
    print("✅ AEC Shared Object Loaded")
except Exception as e:
    print(f"⚠️ AEC Core Failed: {e}")
    aec_state = None

# ---------------------------
# 3. LINUX VOLUME CONTROL (ALSA)
# ---------------------------
def set_linux_volume(percent):
    """Sets system volume using amixer."""
    try:
        # 'Master' is the default. On some Pi DACs, it might be 'Headphone' or 'PCM'
        subprocess.run(["amixer", "sset", "Master", f"{percent}%"], capture_output=True)
    except Exception as e:
        print(f"Volume error: {e}")

# ---------------------------
# 4. VAD SETUP (Optimized)
# ---------------------------
# For Pi 3A, I highly recommend 'pip install onnxruntime' and using the .onnx model
# But for now, here is the torch implementation adapted for Linux:
import torch
model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)

last_vad_prob = 0.0
is_ducked = False
ref_buffer = np.zeros(FRAME_SIZE, dtype=np.float32)

def duck_volume():
    global is_ducked
    is_ducked = True
    set_linux_volume(20) # Duck down to 20%
    time.sleep(2.5)      # Wait for user to finish speaking
    set_linux_volume(80) # Back to normal
    is_ducked = False

# ---------------------------
# 5. AUDIO CALLBACKS
# ---------------------------
def ref_callback(indata, frames, time_info, status):
    global ref_buffer
    ref_buffer[:] = indata[:, 0].astype(np.float32)

def mic_callback(indata, frames, time_info, status):
    global last_vad_prob, is_ducked
    mic_raw = indata[:, 0].astype(np.float32)
    cleaned = np.zeros(FRAME_SIZE, dtype=np.float32)

    if aec_state:
        aec_lib.aec_process_buffer(aec_state, ref_buffer, mic_raw, cleaned, FRAME_SIZE, 0)
    else:
        cleaned = mic_raw

    audio_tensor = torch.from_numpy(cleaned)
    prob = float(model(audio_tensor, SAMPLE_RATE).item())
    last_vad_prob = prob

    if prob > 0.45 and not is_ducked:
        threading.Thread(target=duck_volume, daemon=True).start()

# ---------------------------
# 6. GUI
# ---------------------------
class DuckingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AEC + VAD (Linux/Pi)")
        self.root.geometry("300x150")
        self.running = False
        
        self.btn = ttk.Button(root, text="Start Conversation Mode", command=self.toggle)
        self.btn.pack(pady=20)
        self.prob_lbl = ttk.Label(root, text="Speech Prob: 0%")
        self.prob_lbl.pack()
        self.update_ui()

    def toggle(self):
        if not self.running:
            self.mic_stream = sd.InputStream(device=MIC_DEVICE_ID, channels=1, samplerate=SAMPLE_RATE, callback=mic_callback, blocksize=FRAME_SIZE)
            self.ref_stream = sd.InputStream(device=REF_DEVICE_ID, channels=1, samplerate=SAMPLE_RATE, callback=ref_callback, blocksize=FRAME_SIZE)
            self.mic_stream.start(); self.ref_stream.start()
            self.btn.config(text="Stop")
            self.running = True
        else:
            self.mic_stream.stop(); self.ref_stream.stop()
            self.btn.config(text="Start")
            self.running = False

    def update_ui(self):
        self.prob_lbl.config(text=f"Speech Prob: {int(last_vad_prob * 100)}%")
        self.root.after(100, self.update_ui)

if __name__ == "__main__":
    root = tk.Tk()
    app = DuckingApp(root)
    root.mainloop()
