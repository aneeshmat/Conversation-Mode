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
MIC_ID = 2
LOOP_IDS = [3, 4]

class AppState:
    def __init__(self):
        self.running = True
        self.enabled = True
        self.ducked = False
        self.prob = 0.0
        self.aec_delay = 14  # Increased default for BT
        self.aec_strength = 0.85
        self.vad_threshold = 0.70
        self.duck_vol = 20
        self.norm_vol = 80
        self.unduck_sec = 2.0

state = AppState()

def set_volume(vol_percent):
    try:
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{int(vol_percent)}%"],
                       check=True, capture_output=True)
    except: pass

class SileroVADStateful:
    def __init__(self, onnx_path, sr):
        self.sr = np.array([sr], dtype=np.int64)
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
        # Defensive Input Mapping
        inputs = {i.name: i for i in self.sess.get_inputs()}
        self.name_audio = next((n for n in inputs if "audio" in n or "input" in n or "x" in n), None)
        self.name_sr = next((n for n in inputs if "sr" in n or "sample_rate" in n), None)
        self.name_h = next((n for n in inputs if "h" in n and "n" not in n), None)
        self.name_c = next((n for n in inputs if "c" in n and "n" not in n), None)
        
        outputs = {o.name: o for o in self.sess.get_outputs()}
        self.name_prob = next((n for n in outputs if "prob" in n or "output" in n or "y" in n), None)
        self.name_hn = next((n for n in outputs if "h" in n and n != self.name_h), None)
        self.name_cn = next((n for n in outputs if "c" in n and n != self.name_c), None)

        self.h = np.zeros((2, 1, 64), dtype=np.float32)
        self.c = np.zeros((2, 1, 64), dtype=np.float32)

    def forward(self, audio_16k):
        feed = {self.name_audio: audio_16k.reshape(1, -1).astype(np.float32)}
        if self.name_sr: feed[self.name_sr] = self.sr
        if self.name_h is not None: feed[self.name_h] = self.h
        if self.name_c is not None: feed[self.name_c] = self.c
        
        outs = self.sess.run(None, feed)
        out_map = {o.name: outs[i] for i, o in enumerate(self.sess.get_outputs())}
        
        if self.name_hn: self.h = out_map[self.name_hn]
        if self.name_cn: self.c = out_map[self.name_cn]
        return float(out_map[self.name_prob])

def audio_thread_func():
    # Bluetooth relay typically runs at 44100 or 48000
    SAMPLING_RATE = 48000 
    loop_id, loop_ch = 3, 2
    for lid in LOOP_IDS:
        try:
            dev = sd.query_devices(lid)
            loop_ch, loop_id = min(dev["max_input_channels"], 2), lid
            break
        except: continue

    vad = SileroVADStateful(ONNX_PATH, VAD_SAMPLE_RATE)
    vad_hop = VAD_FRAME_16K
    native_hop = int(vad_hop * (SAMPLING_RATE / VAD_SAMPLE_RATE))
    
    loop_history = deque(maxlen=40) 
    last_speech_time = 0.0
    set_volume(state.norm_vol)

    try:
        with sd.InputStream(device=MIC_ID, channels=1, samplerate=SAMPLING_RATE, blocksize=native_hop) as m_in, \
             sd.InputStream(device=loop_id, channels=loop_ch, samplerate=SAMPLING_RATE, blocksize=native_hop) as l_in:
            
            while state.running:
                mic_chunk, _ = m_in.read(native_hop)
                loop_chunk, _ = l_in.read(native_hop)
                
                # Resample to 16k
                m16 = np.interp(np.linspace(0,1,vad_hop), np.linspace(0,1,native_hop), mic_chunk[:,0]).astype(np.float32)
                l16 = np.interp(np.linspace(0,1,vad_hop), np.linspace(0,1,native_hop), loop_chunk[:,0]).astype(np.float32)

                loop_history.append(l16)
                
                if len(loop_history) >= state.aec_delay:
                    ref = list(loop_history)[-int(state.aec_delay)]
                    clean = m16 - (state.aec_strength * ref)
                    clean = np.clip(clean, -1.0, 1.0)
                else:
                    clean = m16
                
                state.prob = vad.forward(clean)
                now = time.time()

                if state.enabled:
                    if state.prob > state.vad_threshold:
                        if not state.ducked:
                            set_volume(state.duck_vol)
                            state.ducked = True
                        last_speech_time = now
                    elif state.ducked and (now - last_speech_time > state.unduck_sec):
                        set_volume(state.norm_vol)
                        state.ducked = False
                elif state.ducked:
                    set_volume(state.norm_vol)
                    state.ducked = False
    except Exception as e:
        print(f"Audio Error: {e}")

class AudioGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🛡️ Voice Ducking")
        self.root.geometry("400x550")
        self.root.configure(bg="#121212")

        self.btn_toggle = tk.Button(root, text="DISABLE DUCKING", bg="#d32f2f", fg="white", 
                                    font=("Arial", 12, "bold"), command=self.toggle_system)
        self.btn_toggle.pack(fill="x", padx=20, pady=20)

        self.lbl_prob = tk.Label(root, text="Prob: 0.00", bg="#121212", fg="#00ff00")
        self.lbl_prob.pack()
        
        self.prob_bar = ttk.Progressbar(root, length=300, mode='determinate')
        self.prob_bar.pack(pady=10)

        self.create_slider("AEC Delay", 1, 30, "aec_delay")
        self.create_slider("AEC Strength", 0, 1, "aec_strength", True)
        self.create_slider("VAD Threshold", 0, 1, "vad_threshold", True)
        self.create_slider("Normal Volume", 0, 100, "norm_vol")
        
        self.update_gui()

    def toggle_system(self):
        state.enabled = not state.enabled
        self.btn_toggle.config(text="DISABLE DUCKING" if state.enabled else "ENABLE DUCKING",
                               bg="#d32f2f" if state.enabled else "#388e3c")

    def create_slider(self, label, start, end, attr, is_float=False):
        tk.Label(self.root, text=label, bg="#121212", fg="white").pack()
        val = tk.DoubleVar(value=getattr(state, attr))
        def update_val(v): setattr(state, attr, float(v) if is_float else int(float(v)))
        ttk.Scale(self.root, from_=start, to=end, variable=val, command=update_val, orient="horizontal").pack(fill="x", padx=40, pady=5)

    def update_gui(self):
        self.prob_bar['value'] = state.prob * 100
        self.lbl_prob.config(text=f"Speech Probability: {state.prob:.2f}")
        if state.running: self.root.after(100, self.update_gui)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioGUI(root)
    threading.Thread(target=audio_thread_func, daemon=True).start()
    root.mainloop()