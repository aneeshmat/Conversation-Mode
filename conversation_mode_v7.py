# GUI updated, AEC added, and various optimizations. Still no VAD smoothing or hysteresis, but it's a solid base for testing and tuning. The GUI now also provides real-time feedback on the VAD probability and ducking status, making it easier to find the right settings.
import os
import time
import threading
import subprocess
from collections import deque
import tkinter as tk
from tkinter import ttk

import numpy as np
import sounddevice as sd
import onnxruntime as ort

os.environ["ORT_LOGGING_LEVEL"] = "3"

# ----------------- SETTINGS & GLOBALS -----------------
ONNX_PATH = "silero_vad.onnx"
VAD_SAMPLE_RATE = 16000
VAD_FRAME_16K = 512
PREFERRED_OPEN_RATE = 48000
MIC_ID = 2
LOOP_IDS = [3, 4]

class AppState:
    def __init__(self):
        self.running = True
        self.enabled = True  # The new Enable/Disable toggle
        self.ducked = False
        self.prob = 0.0
        # Tuning Settings
        self.aec_delay = 12
        self.aec_strength = 0.85
        self.vad_threshold = 0.70
        self.duck_vol = 20
        self.norm_vol = 80
        self.unduck_sec = 2.0

state = AppState()

# ----------------- AUDIO ENGINE -----------------

def set_volume(vol_percent):
    """Targets Bluetooth Speaker via PulseAudio."""
    try:
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{int(vol_percent)}%"],
                       check=True, capture_output=True)
    except: pass

class SileroVADStateful:
    def __init__(self, onnx_path, sr):
        self.sr = int(sr)
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.name_audio = next(n for n in self.input_names if any(x in n.lower() for x in ["input", "x", "audio"]))
        self.name_sr = next((n for n in self.input_names if any(x in n.lower() for x in ["sr", "sample_rate"])), None)
        self.name_h = next(n for n in self.input_names if "h" in n.lower())
        self.name_c = next(n for n in self.input_names if "c" in n.lower())
        self.name_prob = next(n for n in self.output_names if any(x in n.lower() for x in ["prob", "output", "y"]))
        self.name_hn = next(n for n in self.output_names if "h" in n.lower() and n != self.name_h)
        self.name_cn = next(n for n in self.output_names if "c" in n.lower() and n != self.name_c)
        self.h = np.zeros((2, 1, 64), dtype=np.float32)
        self.c = np.zeros((2, 1, 64), dtype=np.float32)

    def forward(self, audio_16k):
        feed = {self.name_audio: audio_16k.reshape(1, -1).astype(np.float32),
                self.name_h: self.h, self.name_c: self.c}
        if self.name_sr: feed[self.name_sr] = np.array([self.sr], dtype=np.int64)
        outs = self.sess.run(None, feed)
        name2val = {self.output_names[i]: outs[i] for i in range(len(self.output_names))}
        self.h, self.c = name2val[self.name_hn], name2val[self.name_cn]
        return float(np.squeeze(name2val[self.name_prob]))

def audio_thread_func():
    # Setup devices
    mic_rate = 48000 # Default fallback
    loop_id, loop_ch, loop_rate = 3, 2, 48000
    
    # Discovery
    for lid in LOOP_IDS:
        try:
            dev = sd.query_devices(lid)
            loop_ch, loop_id = min(dev["max_input_channels"], 2), lid
            break
        except: continue

    vad = SileroVADStateful(ONNX_PATH, VAD_SAMPLE_RATE)
    vad_hop = VAD_FRAME_16K
    mic_hop = int(round(vad_hop * (48000 / VAD_SAMPLE_RATE))) # Simple static hop for rpi
    loop_hop = int(round(vad_hop * (48000 / VAD_SAMPLE_RATE)))
    
    loop_history = deque(maxlen=40) 
    last_speech_time = 0.0

    try:
        with sd.InputStream(device=MIC_ID, channels=1, samplerate=48000, blocksize=mic_hop) as m_in, \
             sd.InputStream(device=loop_id, channels=loop_ch, samplerate=48000, blocksize=loop_hop) as l_in:
            
            while state.running:
                mic_chunk, _ = m_in.read(mic_hop)
                loop_chunk, _ = l_in.read(loop_hop)
                
                # Resample logic
                m16 = np.interp(np.linspace(0,1,vad_hop), np.linspace(0,1,mic_hop), mic_chunk[:,0]).reshape(-1,1)
                l16 = np.interp(np.linspace(0,1,vad_hop), np.linspace(0,1,loop_hop), loop_chunk[:,0]).reshape(-1,1)

                loop_history.append(l16)
                
                # AEC Logic
                if len(loop_history) >= state.aec_delay:
                    ref = list(loop_history)[-state.aec_delay]
                    clean = m16 - (state.aec_strength * ref)
                    clean = np.clip(clean, -1.0, 1.0)
                else:
                    clean = m16
                
                state.prob = vad.forward(clean)
                now = time.time()

                # DUCKING LOGIC (Only runs if state.enabled is True)
                if state.enabled:
                    if state.prob > state.vad_threshold:
                        if not state.ducked:
                            set_volume(state.duck_vol)
                            state.ducked = True
                        last_speech_time = now
                    elif state.ducked and (now - last_speech_time > state.unduck_sec):
                        set_volume(state.norm_vol)
                        state.ducked = False
                else:
                    # If we just disabled it while ducked, reset volume
                    if state.ducked:
                        set_volume(state.norm_vol)
                        state.ducked = False

    except Exception as e:
        print(f"Audio Error: {e}")

# ----------------- GUI CODE -----------------

class AudioGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🛡️ Ducking Controller")
        self.root.geometry("420x600")
        self.root.configure(bg="#121212")

        # Custom Styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TProgressbar", thickness=20)

        # Enable/Disable Toggle
        self.btn_toggle = tk.Button(root, text="DISABLE DUCKING", font=("Arial", 12, "bold"), 
                                    bg="#d32f2f", fg="white", command=self.toggle_system, pady=10)
        self.btn_toggle.pack(fill="x", padx=20, pady=15)

        # Status Bar
        self.lbl_status = tk.Label(root, text="SYSTEM ENABLED", font=("Arial", 14, "bold"), bg="#333", fg="white")
        self.lbl_status.pack(fill="x", padx=10)

        # Probability Display
        self.lbl_prob = tk.Label(root, text="Speech Prob: 0.00", bg="#121212", fg="#00ff00", font=("Courier", 12))
        self.lbl_prob.pack(pady=(15,0))
        
        self.prob_bar = ttk.Progressbar(root, length=300, mode='determinate')
        self.prob_bar.pack(pady=5)

        # Sliders
        self.create_slider("AEC Delay (Bluetooth Lag)", 1, 30, "aec_delay")
        self.create_slider("AEC Strength", 0, 1, "aec_strength", is_float=True)
        self.create_slider("VAD Threshold", 0, 1, "vad_threshold", is_float=True)
        self.create_slider("Normal Volume %", 0, 100, "norm_vol")
        self.create_slider("Duck Volume %", 0, 100, "duck_vol")

        self.update_gui()

    def toggle_system(self):
        state.enabled = not state.enabled
        if state.enabled:
            self.btn_toggle.config(text="DISABLE DUCKING", bg="#d32f2f")
            self.lbl_status.config(text="SYSTEM ENABLED", bg="#333")
        else:
            self.btn_toggle.config(text="ENABLE DUCKING", bg="#388e3c")
            self.lbl_status.config(text="SYSTEM DISABLED (BYPASS)", bg="#222")

    def create_slider(self, label, start, end, attr, is_float=False):
        frame = tk.Frame(self.root, bg="#121212")
        frame.pack(fill="x", padx=40, pady=5)
        
        tk.Label(frame, text=label, bg="#121212", fg="gray", font=("Arial", 10)).pack(side="left")
        self.val_label = tk.Label(frame, text="", bg="#121212", fg="white")
        self.val_label.pack(side="right")
        
        val = tk.DoubleVar(value=getattr(state, attr))
        
        def update_val(v):
            new_val = float(v) if is_float else int(float(v))
            setattr(state, attr, new_val)
            
        scale = ttk.Scale(self.root, from_=start, to=end, variable=val, command=update_val, orient="horizontal")
        scale.pack(fill="x", padx=40, pady=(0, 10))

    def update_gui(self):
        # Update UI Elements
        self.prob_bar['value'] = state.prob * 100
        self.lbl_prob.config(text=f"Speech Prob: {state.prob:.2f}")

        # Visual feedback for ducking activity
        if state.enabled:
            if state.ducked:
                self.lbl_status.config(bg="#d32f2f", text="DUCKING ACTIVE")
            else:
                self.lbl_status.config(bg="#388e3c", text="MONITORING SPEECH")
        
        if state.running:
            self.root.after(100, self.update_gui)

def on_closing():
    state.running = False
    set_volume(state.norm_vol)
    root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    app = AudioGUI(root)
    
    t = threading.Thread(target=audio_thread_func, daemon=True)
    t.start()
    
    root.mainloop()