#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conversation Mode with Silero VAD (stateful ONNX)
- Optimized for stability and sample-rate safety.
"""

import os
import time
import shutil
import subprocess
from collections import deque

import numpy as np
import sounddevice as sd
import alsaaudio
import onnxruntime as ort

# Set ONNX logging BEFORE importing onnxruntime
os.environ["ORT_LOGGING_LEVEL"] = "3"

# ----------------- USER SETTINGS -----------------
MIC_ID = 2
LOOP_IDS = [3, 4] 
ONNX_PATH = "silero_vad.onnx"

VAD_SAMPLE_RATE = 16000
VAD_FRAME_16K = 512 
PREFERRED_OPEN_RATE = 48000

AEC_DELAY_FRAMES_16K = 3
VAD_THRESHOLD = 0.65
UNDUCK_AFTER_SEC = 2.5
DUCK_VOLUME = 20
NORMAL_VOLUME = 80

# ----------------- HELPERS -----------------

def nearest_working_samplerate(dev_index: int, desired: int, channels: int = 1) -> int:
    dev = sd.query_devices(dev_index)
    candidates = [int(desired)] if desired else []
    default_rate = int(dev.get("default_samplerate") or 0)
    if default_rate: candidates.append(default_rate)
    for r in (48000, 44100, 32000, 16000):
        if r not in candidates: candidates.append(r)

    for rate in candidates:
        try:
            sd.check_input_settings(device=dev_index, samplerate=rate, channels=channels)
            return rate
        except Exception:
            continue
    raise RuntimeError(f"No workable samplerate for device #{dev_index}")

def init_mixer(preferred_controls=("PCM", "Master", "Speaker", "Playback")):
    try:
        cards = alsaaudio.cards()
        for ci, _ in enumerate(cards):
            try:
                ctrls = alsaaudio.mixers(ci)
                for ctrl in preferred_controls:
                    if ctrl in ctrls:
                        return alsaaudio.Mixer(control=ctrl, cardindex=ci)
            except Exception:
                continue
    except Exception:
        pass
    return None

def set_volume(mixer, vol_percent: int):
    if mixer is not None:
        try:
            mixer.setvolume(int(vol_percent))
            return
        except Exception:
            pass

    if shutil.which("wpctl"):
        try:
            scalar = max(0.0, min(1.0, vol_percent / 100.0))
            subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{scalar:.2f}"],
                           check=False, capture_output=True)
            return
        except Exception:
            pass

    if shutil.which("pactl"):
        try:
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{int(vol_percent)}%"],
                           check=False, capture_output=True)
            return
        except Exception:
            pass

# -------- Silero VAD (stateful) session wrapper --------

class SileroVADStateful:
    def __init__(self, onnx_path: str, sr: int):
        self.sr = int(sr)
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.in_map, self.out_map = self._discover_io(self.sess)
        self.h = self._zeros_like_input(self.in_map["h"])
        self.c = self._zeros_like_input(self.in_map["c"])

    def _discover_io(self, sess):
        inputs = sess.get_inputs()
        in_audio = in_sr = in_h = in_c = None
        
        for i in inputs:
            nm = i.name.lower()
            if i.type.startswith("tensor(int"): in_sr = i
            elif any(x in nm for x in ["input", "x", "audio"]): in_audio = i
            elif "h" in nm: in_h = i
            elif "c" in nm: in_c = i

        outputs = sess.get_outputs()
        out_prob = out_hn = out_cn = None
        for o in outputs:
            nm = o.name.lower()
            if any(x in nm for x in ["prob", "output", "y"]): out_prob = o
            elif any(x in nm for x in ["hn", "h1"]) or nm == "h": out_hn = o
            elif any(x in nm for x in ["cn", "c1"]) or nm == "c": out_cn = o

        return {"audio": in_audio, "h": in_h, "c": in_c, "sr": in_sr}, \
               {"prob": out_prob, "hn": out_hn, "cn": out_cn}

    def _zeros_like_input(self, meta):
        shp = meta.shape
        if shp and all(isinstance(d, int) for d in shp):
            return np.zeros(tuple(shp), dtype=np.float32)
        return np.zeros((2, 1, 64), dtype=np.float32)

    def forward(self, audio_16k: np.ndarray) -> float:
        x = audio_16k.reshape(1, -1).astype(np.float32)
        sr = np.array([self.sr], dtype=np.int64)
        feed = {
            self.in_map["audio"].name: x,
            self.in_map["h"].name: self.h,
            self.in_map["c"].name: self.c,
            self.in_map["sr"].name: sr
        }
        outs = self.sess.run(None, feed)
        names = [o.name for o in self.sess.get_outputs()]
        name2val = {names[i]: outs[i] for i in range(len(names))}
        
        prob = float(np.squeeze(name2val[self.out_map["prob"].name]))
        self.h = name2val[self.out_map["hn"].name]
        self.c = name2val[self.out_map["cn"].name]
        return prob

def linear_resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or x.size == 0:
        return x.astype(np.float32, copy=False)
    n_src = x.shape[0]
    n_dst = int(round(n_src * (dst_sr / float(src_sr))))
    if n_dst <= 1: return np.zeros((n_dst, x.shape[1]), dtype=np.float32)
    t_src = np.linspace(0.0, 1.0, n_src, endpoint=False)
    t_dst = np.linspace(0.0, 1.0, n_dst, endpoint=False)
    y = np.empty((n_dst, x.shape[1]), dtype=np.float32)
    for c in range(x.shape[1]):
        y[:, c] = np.interp(t_dst, t_src, x[:, c])
    return y

def choose_loop_device(loop_ids):
    for lid in loop_ids:
        try:
            dev = sd.query_devices(lid)
            ch = 1 if dev["max_input_channels"] <= 1 else min(dev["max_input_channels"], 2)
            rate = nearest_working_samplerate(lid, PREFERRED_OPEN_RATE, channels=ch)
            return lid, dev, ch, rate
        except Exception:
            continue
    raise RuntimeError("No loopback device usable.")

# ----------------- MAIN -----------------

def main():
    mic_rate = nearest_working_samplerate(MIC_ID, PREFERRED_OPEN_RATE, channels=1)
    loop_id, loop_dev, loop_ch, loop_rate = choose_loop_device(LOOP_IDS)

    mixer = init_mixer()
    set_volume(mixer, NORMAL_VOLUME)
    vad = SileroVADStateful(ONNX_PATH, VAD_SAMPLE_RATE)

    vad_hop_16k = VAD_FRAME_16K
    mic_hop_native = int(round(vad_hop_16k * (mic_rate / VAD_SAMPLE_RATE)))
    loop_hop_native = int(round(vad_hop_16k * (loop_rate / VAD_SAMPLE_RATE)))
    aec_delay_frames = max(1, int(AEC_DELAY_FRAMES_16K))

    loop_history_16k = deque(maxlen=max(aec_delay_frames + 8, 16))
    ducked = False
    last_speech_time = 0.0

    mic_accum = np.zeros((0, 1), dtype=np.float32)
    loop_accum = np.zeros((0, 1), dtype=np.float32)

    print("🚀 Audio Processing Started. Press Ctrl+C to stop.")

    try:
        with sd.InputStream(device=MIC_ID, channels=1, samplerate=mic_rate, blocksize=mic_hop_native) as mic_in, \
             sd.InputStream(device=loop_id, channels=loop_ch, samplerate=loop_rate, blocksize=loop_hop_native) as loop_in:

            while True:
                # Use non-blocking checks to prevent hanging
                mic_chunk, _ = mic_in.read(mic_hop_native)
                loop_chunk_multi, _ = loop_in.read(loop_hop_native)

                if mic_chunk is None or loop_chunk_multi is None:
                    continue

                # Ensure 2D and handle loopback channels
                mic_chunk = mic_chunk.reshape(-1, 1).astype(np.float32)
                loop_chunk = loop_chunk_multi[:, 0:1].astype(np.float32)

                mic_accum = np.vstack((mic_accum, mic_chunk))
                loop_accum = np.vstack((loop_accum, loop_chunk))

                # Process if we have enough to resample
                if mic_accum.shape[0] >= mic_hop_native and loop_accum.shape[0] >= loop_hop_native:
                    mic_frame_16k = linear_resample(mic_accum[:mic_hop_native], mic_rate, VAD_SAMPLE_RATE)
                    loop_frame_16k = linear_resample(loop_accum[:loop_hop_native], loop_rate, VAD_SAMPLE_RATE)
                    
                    # Trim accumulators
                    mic_accum = mic_accum[mic_hop_native:]
                    loop_accum = loop_accum[loop_hop_native:]

                    # --- AEC ---
                    loop_history_16k.append(loop_frame_16k.copy())
                    if len(loop_history_16k) > aec_delay_frames:
                        delayed = loop_history_16k[-(aec_delay_frames + 1)]
                    else:
                        delayed = np.zeros_like(loop_frame_16k)

                    clean_16k = mic_frame_16k - 0.7 * delayed
                    clean_16k = np.clip(clean_16k, -1.0, 1.0)

                    # --- VAD & Ducking ---
                    prob = vad.forward(clean_16k)
                    now = time.time()
                    if prob > VAD_THRESHOLD:
                        if not ducked:
                            print(f"🎤 Speech ({int(prob*100)}%) -> Ducking")
                            set_volume(mixer, DUCK_VOLUME)
                            ducked = True
                        last_speech_time = now
                    elif ducked and (now - last_speech_time > UNDUCK_AFTER_SEC):
                        print("🔇 Silence -> Restoring")
                        set_volume(mixer, NORMAL_VOLUME)
                        ducked = False

    except KeyboardInterrupt:
        print("\n👋 Stopped.")
    except Exception as e:
        print(f"\nError: {e}")
        raise

if __name__ == "__main__":
    main()