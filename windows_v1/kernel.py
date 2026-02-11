import time
import threading
import torch
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import messagebox, ttk
import ctypes
from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ---------------------------
# 1. HARDCODED HARDWARE SPECS
# ---------------------------
# IDs from your query_devices output
MIC_ID = 30
MIC_RATE = 48000
MIC_CHANNELS = 1

WILLEN_ID = 26
WILLEN_RATE = 44100
WILLEN_CHANNELS = 2

FRAME_SIZE = 512
DUCK_RATIO = 0.50

# ---------------------------
# 2. GLOBAL BUFFERS & STATE
# ---------------------------
# Shared buffer for AEC reference
ref_frame = np.zeros(FRAME_SIZE, dtype=np.float32)
buffer_lock = threading.Lock()
last_vad_prob = 0.0
is_running = False

# Load VAD
model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)

# Load AEC DLL
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
except:
    aec_state = None

# Volume Control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))

# ---------------------------
# 3. THE THREADED WORKERS
# ---------------------------

def loopback_worker():
    """Captures the Willen audio as a reference for AEC."""
    global ref_frame, is_running
    wasapi_settings = sd.WasapiSettings()
    wasapi_settings.loopback = True
    
    try:
        with sd.InputStream(device=WILLEN_ID, 
                            channels=WILLEN_CHANNELS, 
                            samplerate=WILLEN_RATE,
                            blocksize=FRAME_SIZE,
                            extra_settings=wasapi_settings) as stream:
            while is_running:
                data, overflowed = stream.read(FRAME_SIZE)
                with buffer_lock:
                    # Capture channel 0 (Left) for mono AEC reference
                    ref_frame[:] = data[:, 0].astype(np.float32)
    except Exception as e:
        print(f"Loopback Thread Error: {e}")

def mic_worker():
    """Captures Mic, cleans with AEC, and runs VAD."""
    global last_vad_prob, is_running
    try:
        with sd.InputStream(device=MIC_ID, 
                            channels=MIC_CHANNELS, 
                            samplerate=MIC_RATE,
                            blocksize=FRAME_SIZE) as stream:
            while is_running:
                mic_data, overflowed = stream.read(FRAME_SIZE)
                
                with buffer_lock:
                    current_ref = ref_frame.copy()

                mic_float = mic_data[:, 0].astype(np.float32)
                cleaned = np.zeros(FRAME_SIZE, dtype=np.float32)

                if aec_state:
                    # Use a high delay for Bluetooth latency calibration
                    aec_lib.aec_process_buffer(aec_state, current_ref, mic_float, cleaned, FRAME_SIZE, 2400)
                else:
                    cleaned = mic_float

                # VAD Processing
                audio_tensor = torch.from_numpy(cleaned)
                prob = float(model(audio_tensor, MIC_RATE).item())
                last_vad_prob = prob

                # Ducking Logic
                if prob > 0.45:
                    curr = volume_ctrl.GetMasterVolumeLevelScalar()
                    volume_ctrl.SetMasterVolumeLevelScalar(curr * DUCK_RATIO, None)
                    # Use a non-blocking timer to restore volume
                    threading.Timer(2.5, lambda c=curr: volume_ctrl.SetMasterVolumeLevelScalar(c, None)).start()

    except Exception as e:
        print(f"Mic Thread Error: {e}")

# ---------------------------
# 4. GUI INTERFACE
# ---------------------------
class ConversationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AEC v9.0 - Multi-Threaded")
        self.enabled = False
        
        self.btn = ttk.Button(root, text="Enable Conversation Mode", command=self.toggle)
        self.btn.pack(pady=20, padx=40)
        
        self.status_var = tk.StringVar(value="VAD: 0%")
        ttk.Label(root, textvariable=self.status_var).pack(pady=10)
        
        self.update_gui()

    def toggle(self):
        global is_running
        if not self.enabled:
            is_running = True
            self.enabled = True
            self.btn.config(text="Disable")
            
            # Start threads
            threading.Thread(target=loopback_worker, daemon=True).start()
            threading.Thread(target=mic_worker, daemon=True).start()
        else:
            is_running = False
            self.enabled = False
            self.btn.config(text="Enable")

    def update_gui(self):
        self.status_var.set(f"VAD Probability: {int(last_vad_prob * 100)}%")
        self.root.after(100, self.update_gui)

if __name__ == "__main__":
    root = tk.Tk()
    app = ConversationApp(root)
    root.mainloop()