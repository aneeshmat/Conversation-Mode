import time
import threading
import torch
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import messagebox, ttk
import ctypes
import queue

# pycaw imports for Windows volume control
from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# --- 1. CONFIGURATION ---
MIC_ID = 1          # Microphone (USB PnP Audio Device)
LOOPBACK_ID = 3     # CABLE Output (VB-Audio Virtual Cable)
SAMPLE_RATE = 16000
FRAME_SIZE = 512
AEC_DELAY = 160     # Adjust based on loopback latency

# Queues for cross-stream synchronization
speaker_q = queue.Queue(maxsize=20)

# --- 2. AEC DLL SETUP ---
try:
    aec_lib = ctypes.CDLL('./aec_vad.dll')
    aec_lib.aec_create.restype = ctypes.c_void_p
    aec_lib.aec_process_buffer.argtypes = [
        ctypes.c_void_p, 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float), 
        ctypes.c_int, ctypes.c_int
    ]
    aec_lib.aec_free.argtypes = [ctypes.c_void_p]
    aec_state = aec_lib.aec_create()
except Exception as e:
    print(f"DLL Error: {e}")
    exit()

# --- 3. VAD MODEL ---
model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

# --- 4. VOLUME CONTROL ---
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Ducking Params
DUCK_RATIO = 0.50
MIN_BASELINE = 0.10
duck_duration = 2
is_ducked = False
pre_duck_volume = None
duck_timer = None
duck_lock = threading.Lock()

# Metrics
last_vad_prob = 0.0
last_is_speech = False
prev_state = 'SILENCE'

# --- 5. VOLUME LOGIC ---
def _set_vol(val):
    try: volume.SetMasterVolumeLevelScalar(max(0.0, min(1.0, val)), None)
    except: pass

def restore_volume():
    global is_ducked, pre_duck_volume
    with duck_lock:
        if is_ducked and pre_duck_volume:
            _set_vol(pre_duck_volume)
            is_ducked = False
            print("🔊 Volume Restored")

def duck_volume():
    global is_ducked, pre_duck_volume, duck_timer
    with duck_lock:
        current = volume.GetMasterVolumeLevelScalar()
        if current < MIN_BASELINE: return
        
        if not is_ducked:
            pre_duck_volume = current
            _set_vol(current * DUCK_RATIO)
            is_ducked = True
            print("⤵️ Ducking Active")
        
        if duck_timer: duck_timer.cancel()
        duck_timer = threading.Timer(duck_duration, restore_volume)
        duck_timer.start()

# --- 6. AUDIO PIPELINE CALLBACKS ---

def speaker_callback(indata, frames, time, status):
    """Captures system audio to use as a reference for AEC."""
    # indata is mono/stereo from VB-Cable; take first channel
    data = indata[:, 0].astype(np.float32)
    try:
        speaker_q.put_nowait(data)
    except queue.Full:
        speaker_q.get()
        speaker_q.put_nowait(data)

def mic_callback(indata, frames, time_info, status):
    """Captures mic, cleans it with AEC using speaker_q, then runs VAD."""
    global last_vad_prob, last_is_speech, prev_state

    # 1. Get Speaker Ref
    try:
        ref_chunk = speaker_q.get_nowait()
    except queue.Empty:
        ref_chunk = np.zeros(frames, dtype=np.float32)

    # 2. AEC Processing
    mic_chunk = indata[:, 0].astype(np.float32)
    cleaned_mic = np.zeros(frames, dtype=np.float32)
    
    aec_lib.aec_process_buffer(
        aec_state,
        ref_chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        mic_chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        cleaned_mic.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        frames,
        AEC_DELAY
    )

    # 3. VAD on Cleaned Audio
    audio_tensor = torch.from_numpy(cleaned_mic)
    prob = model(audio_tensor, SAMPLE_RATE).item()
    last_vad_prob = prob
    
    is_speech = prob > 0.4
    last_is_speech = is_speech

    if is_speech and prev_state != 'SPEAKING':
        prev_state = 'SPEAKING'
        duck_volume()
    elif not is_speech and prev_state == 'SPEAKING':
        prev_state = 'SILENCE'

# --- 7. GUI ---
class DuckingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AEC + VAD Pipeline")
        self.streams = []
        
        frame = ttk.Frame(root, padding=20)
        frame.pack()
        
        self.btn = ttk.Button(frame, text="Start Pipeline", command=self.toggle)
        self.btn.pack()
        
        self.label = ttk.Label(frame, text="VAD Prob: 0%", font=("Arial", 12))
        self.label.pack(pady=10)
        
        self.update_loop()

    def toggle(self):
        if not self.streams:
            try:
                # Start Speaker Loopback Stream
                s1 = sd.InputStream(device=LOOPBACK_ID, channels=1, callback=speaker_callback, 
                                    samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE)
                # Start Mic Stream
                s2 = sd.InputStream(device=MIC_ID, channels=1, callback=mic_callback, 
                                    samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE)
                s1.start(); s2.start()
                self.streams = [s1, s2]
                self.btn.config(text="Stop Pipeline")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            for s in self.streams: s.stop(); s.close()
            self.streams = []
            self.btn.config(text="Start Pipeline")
            restore_volume()

    def update_loop(self):
        self.label.config(text=f"VAD Prob: {int(last_vad_prob*100)}% | Speaking: {last_is_speech}")
        self.root.after(100, self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = DuckingApp(root)
    root.mainloop()