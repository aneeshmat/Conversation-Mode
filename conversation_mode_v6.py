# still no AEC
import os
import time
import shutil
import subprocess
from collections import deque

import numpy as np
import sounddevice as sd
import onnxruntime as ort

os.environ["ORT_LOGGING_LEVEL"] = "3"

# ----------------- USER SETTINGS -----------------
MIC_ID = 2
LOOP_IDS = [3, 4]
ONNX_PATH = "silero_vad.onnx"

VAD_SAMPLE_RATE = 16000
VAD_FRAME_16K = 512
PREFERRED_OPEN_RATE = 48000

# --- AEC TUNING ---
# Bluetooth delay is huge. 1 frame = 32ms. 
# Try values between 10 and 20 (320ms - 640ms)
AEC_DELAY_FRAMES_16K = 12 
AEC_STRENGTH = 0.85      # 1.0 is total subtraction, 0.7 is partial

VAD_THRESHOLD = 0.70     # Higher = less sensitive to background noise
UNDUCK_AFTER_SEC = 2.0
DUCK_VOLUME = 20
NORMAL_VOLUME = 80

# ----------------- HELPERS -----------------

def nearest_working_samplerate(dev_index: int, desired: int, channels: int = 1) -> int:
    dev = sd.query_devices(dev_index)
    candidates = [int(desired), 48000, 44100, 16000]
    for rate in candidates:
        try:
            sd.check_input_settings(device=dev_index, samplerate=rate, channels=channels)
            return rate
        except: continue
    raise RuntimeError(f"No workable rate for device #{dev_index}")

def set_volume(vol_percent: int):
    try:
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{int(vol_percent)}%"],
                       check=True, capture_output=True)
    except: pass

# -------- Silero VAD (stateful) --------

class SileroVADStateful:
    def __init__(self, onnx_path: str, sr: int):
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

    def forward(self, audio_16k: np.ndarray) -> float:
        feed = {
            self.name_audio: audio_16k.reshape(1, -1).astype(np.float32),
            self.name_h: self.h,
            self.name_c: self.c
        }
        if self.name_sr:
            feed[self.name_sr] = np.array([self.sr], dtype=np.int64)
        
        outs = self.sess.run(None, feed)
        name2val = {self.output_names[i]: outs[i] for i in range(len(self.output_names))}
        self.h = name2val[self.name_hn]
        self.c = name2val[self.name_cn]
        return float(np.squeeze(name2val[self.name_prob]))

def linear_resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr: return x.astype(np.float32)
    n_dst = int(round(len(x) * (dst_sr / src_sr)))
    t_src = np.linspace(0, 1, len(x), endpoint=False)
    t_dst = np.linspace(0, 1, n_dst, endpoint=False)
    return np.interp(t_dst, t_src, x[:, 0]).reshape(-1, 1).astype(np.float32)

# ----------------- MAIN -----------------

def main():
    print("🛡️ AEC-Tuned Bluetooth Relay Mode - Starting...")
    
    mic_rate = nearest_working_samplerate(MIC_ID, PREFERRED_OPEN_RATE, 1)
    
    loop_id = 3
    loop_ch = 2
    loop_rate = 48000
    for lid in LOOP_IDS:
        try:
            dev = sd.query_devices(lid)
            loop_ch = min(dev["max_input_channels"], 2)
            loop_rate = nearest_working_samplerate(lid, PREFERRED_OPEN_RATE, loop_ch)
            loop_id = lid
            break
        except: continue

    set_volume(NORMAL_VOLUME)
    vad = SileroVADStateful(ONNX_PATH, VAD_SAMPLE_RATE)

    vad_hop_16k = VAD_FRAME_16K
    mic_hop_native = int(round(vad_hop_16k * (mic_rate / VAD_SAMPLE_RATE)))
    loop_hop_native = int(round(vad_hop_16k * (loop_rate / VAD_SAMPLE_RATE)))
    
    # Large buffer to handle Bluetooth latency
    loop_history = deque(maxlen=AEC_DELAY_FRAMES_16K + 2)
    ducked = False
    last_speech = 0.0

    print(f"🚀 AEC Calibrated: Delay={AEC_DELAY_FRAMES_16K} frames (~{AEC_DELAY_FRAMES_16K*32}ms)")

    try:
        with sd.InputStream(device=MIC_ID, channels=1, samplerate=mic_rate, blocksize=mic_hop_native) as m_in, \
             sd.InputStream(device=loop_id, channels=loop_ch, samplerate=loop_rate, blocksize=loop_hop_native) as l_in:
            
            while True:
                mic_chunk, _ = m_in.read(mic_hop_native)
                loop_chunk_m, _ = l_in.read(loop_hop_native)
                
                mic_16k = linear_resample(mic_chunk, mic_rate, VAD_SAMPLE_RATE)
                loop_16k = linear_resample(loop_chunk_m[:, 0:1], loop_rate, VAD_SAMPLE_RATE)

                loop_history.append(loop_16k)
                
                # Wait until the history buffer is full before trying to cancel
                if len(loop_history) >= AEC_DELAY_FRAMES_16K:
                    delayed_reference = loop_history[0]
                    # Subtract music from microphone signal
                    clean = mic_16k - (AEC_STRENGTH * delayed_reference)
                    clean = np.clip(clean, -1.0, 1.0)
                else:
                    clean = mic_16k
                
                prob = vad.forward(clean)
                now = time.time()

                if prob > VAD_THRESHOLD:
                    if not ducked:
                        print(f"🎤 Speech ({int(prob*100)}%) -> Ducking")
                        set_volume(DUCK_VOLUME)
                        ducked = True
                    last_speech = now
                elif ducked and (now - last_speech > UNDUCK_AFTER_SEC):
                    print("🔇 Silence -> Restoring")
                    set_volume(NORMAL_VOLUME)
                    ducked = False

    except KeyboardInterrupt:
        set_volume(NORMAL_VOLUME)

if __name__ == "__main__":
    main()