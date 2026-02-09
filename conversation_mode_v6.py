#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conversation Mode with Silero VAD (stateful ONNX)
- Mic index = 2
- Loopback indices = [3, 4] (fallback: try 3, then 4)
- Opens devices at a supported native rate (e.g., 48 kHz), resamples to 16 kHz for VAD
- Handles stateful Silero VAD: inputs (x/input, h/h1, c/c1, sr) and outputs (prob, hn, cn)
- CPU-only ONNX; robust input/output name discovery
- Auto-detect loopback channels (1 or 2), uses channel 0
- Simple AEC (delayed subtraction) + volume ducking (ALSA → wpctl → pactl fallback)
"""

import os
# Set ONNX logging BEFORE importing onnxruntime
os.environ["ORT_LOGGING_LEVEL"] = "3"

import time
import shutil
import subprocess
from collections import deque

import numpy as np
import sounddevice as sd
import alsaaudio
import onnxruntime as ort

# ----------------- USER SETTINGS -----------------
MIC_ID = 2
LOOP_IDS = [3, 4]       # try 3; if it fails, try 4
ONNX_PATH = "silero_vad.onnx"

# VAD runs at 16 kHz mono
VAD_SAMPLE_RATE = 16000
VAD_FRAME_16K = 512              # ~32 ms frames into VAD

# Open hardware at a common/native rate (we negotiate if needed)
PREFERRED_OPEN_RATE = 48000

# AEC: delay (in VAD frames @16k). Tune 2–6 depending on your system.
AEC_DELAY_FRAMES_16K = 3

# VAD/ducking params
VAD_THRESHOLD = 0.65
UNDUCK_AFTER_SEC = 2.5
DUCK_VOLUME = 20
NORMAL_VOLUME = 80

# ----------------- HELPERS -----------------

def nearest_working_samplerate(dev_index: int, desired: int, channels: int = 1) -> int:
    """Find a samplerate that the device accepts for the given channel count."""
    dev = sd.query_devices(dev_index)
    candidates = []
    if desired: candidates.append(int(desired))
    default_rate = int(dev.get("default_samplerate") or 0)
    if default_rate: candidates.append(default_rate)
    for r in (48000, 44100, 32000, 16000):
        if r not in candidates:
            candidates.append(r)

    for rate in candidates:
        try:
            sd.check_input_settings(device=dev_index, samplerate=rate, channels=channels)
            return rate
        except Exception:
            continue
    raise RuntimeError(f"No workable samplerate for device #{dev_index} with {channels} channels")

def init_mixer(preferred_controls=("PCM", "Master", "Speaker", "Playback")):
    """Find an ALSA mixer control; fall back to wpctl/pactl inside set_volume if None."""
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
    print("⚠️  No suitable ALSA mixer found; will try PipeWire/PulseAudio if available.")
    return None

def set_volume(mixer, vol_percent: int):
    """Set volume using ALSA mixer; fallback to wpctl/pactl if ALSA missing/ineffective."""
    # ALSA
    if mixer is not None:
        try:
            mixer.setvolume(int(vol_percent))
            return
        except Exception as e:
            print(f"ALSA mixer setvolume failed: {e}")

    # PipeWire (wpctl expects 0.0–1.0)
    if shutil.which("wpctl"):
        try:
            scalar = max(0.0, min(1.0, vol_percent / 100.0))
            subprocess.run(
                ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{scalar:.2f}"],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return
        except Exception as e:
            print(f"wpctl volume set failed: {e}")

    # PulseAudio
    if shutil.which("pactl"):
        try:
            subprocess.run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{int(vol_percent)}%"],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return
        except Exception as e:
            print(f"pactl volume set failed: {e}")

    print("⚠️  Could not set volume via ALSA/wpctl/pactl. Ducking may not take effect.")

# -------- Silero VAD (stateful) session wrapper --------

class SileroVADStateful:
    """
    Handles stateful Silero VAD ONNX:
      inputs: x/input (audio 1xN), h/h1, c/c1 (state tensors), sr (int64)
      outputs: prob, hn, cn (names vary)
    Discovers names & shapes at runtime and maintains h/c across frames.
    """
    def __init__(self, onnx_path: str, sr: int):
        self.sr = int(sr)
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.in_map, self.out_map = self._discover_io(self.sess)

        # Initialize h, c zeros using input shapes (or output shapes if inputs are dynamic)
        self.h = self._zeros_like_input(self.in_map["h"])
        self.c = self._zeros_like_input(self.in_map["c"])

    def _discover_io(self, sess):
        # Classify inputs
        in_audio = None
        in_sr = None
        in_h = None
        in_c = None

        # Helper to get rank safely
        def rank(i): 
            shp = i.shape
            try:
                return len(shp) if shp is not None else None
            except Exception:
                return None

        inputs = sess.get_inputs()
        # First pass: by dtype
        for i in inputs:
            if i.type.startswith("tensor(int"):
                in_sr = i
            else:
                # float inputs: could be audio or states
                name = i.name.lower()
                r = rank(i)
                # Name‑based preference
                if in_audio is None and ("input" in name or name == "x" or "audio" in name):
                    in_audio = i
                elif in_h is None and ("h" in name):
                    in_h = i
                elif in_c is None and ("c" in name):
                    in_c = i

        # Second pass: by shape/rank heuristics for anything missing
        float_inputs = [i for i in inputs if not i.type.startswith("tensor(int"))]
        if in_audio is None:
            # pick a float input with rank 2 as audio (1, N) is common
            cand = [i for i in float_inputs if rank(i) == 2]
            in_audio = cand[0] if cand else (float_inputs[0] if float_inputs else None)
        # States: prefer rank 3 (e.g., (2,1,64)); else whatever remains
        remaining = [i for i in float_inputs if i is not in {in_audio}]
        if in_h is None and remaining:
            cand = [i for i in remaining if "h" in i.name.lower()] or [i for i in remaining if rank(i) == 3]
            in_h = cand[0] if cand else remaining[0]
            remaining = [i for i in remaining if i is not in {in_h}]
        if in_c is None and remaining:
            cand = [i for i in remaining if "c" in i.name.lower()] or [i for i in remaining if rank(i) == 3]
            in_c = cand[0] if cand else remaining[0]

        if not (in_audio and in_sr and in_h and in_c):
            names = [i.name for i in inputs]
            raise RuntimeError(f"Could not map VAD inputs; model inputs: {names}")

        # Classify outputs
        outputs = sess.get_outputs()
        out_prob = None
        out_hn = None
        out_cn = None
        for o in outputs:
            nm = o.name.lower()
            if out_prob is None and ("out" in nm or nm in ("y", "prob", "output")):
                out_prob = o
            elif out_hn is None and ("hn" in nm or nm == "h" or "h1" in nm):
                out_hn = o
            elif out_cn is None and ("cn" in nm or nm == "c" or "c1" in nm):
                out_cn = o
        # Fallback: pick by rank (prob scalar → rank 0/1; states → rank ≥2)
        if out_prob is None:
            scalars = [o for o in outputs if o.shape in ([], [1]) or (len(o.shape or []) <= 1)]
            out_prob = scalars[0] if scalars else outputs[0]
        state_like = [o for o in outputs if o is not out_prob]
        if out_hn is None and state_like:
            out_hn = state_like[0]
        if out_cn is None and len(state_like) > 1:
            out_cn = state_like[1]

        # Report mapping
        print(f"VAD ONNX inputs mapped: audio='{in_audio.name}', h='{in_h.name}', c='{in_c.name}', sr='{in_sr.name}'")
        print(f"VAD ONNX outputs mapped: prob='{out_prob.name}', hn='{out_hn.name}', cn='{out_cn.name}'")

        in_map = {"audio": in_audio, "h": in_h, "c": in_c, "sr": in_sr}
        out_map = {"prob": out_prob, "hn": out_hn, "cn": out_cn}
        return in_map, out_map

    def _zeros_like_input(self, meta):
        # Try to use input shape; if dynamic, try output shape; else default to (2,1,64)
        shp = meta.shape
        def to_int(x): 
            try: 
                return int(x)
            except Exception:
                return None
        if shp and all(to_int(d) is not None for d in shp):
            shape = tuple(int(d) for d in shp)
        else:
            # Try outputs (hn/cn) to get concrete shapes
            outs = self.sess.get_outputs()
            for o in outs:
                nm = o.name.lower()
                if ("hn" in nm or nm == "h" or "h1" in nm or "cn" in nm or nm == "c" or "c1" in nm):
                    s = o.shape
                    if s and all(to_int(d) is not None for d in s):
                        shape = tuple(int(d) for d in s)
                        break
            else:
                # Reasonable default for Silero (2 layers, batch 1, hidden 64)
                shape = (2, 1, 64)
        return np.zeros(shape, dtype=np.float32)

    def forward(self, audio_16k: np.ndarray) -> float:
        """
        audio_16k: (N, 1) float32 at 16k
        Returns probability; updates self.h/self.c with recurrent states.
        """
        x = audio_16k.reshape(1, -1).astype(np.float32)
        sr = np.array([self.sr], dtype=np.int64)
        feed = {
            self.in_map["audio"].name: x,
            self.in_map["h"].name: self.h,
            self.in_map["c"].name: self.c,
            self.in_map["sr"].name: sr
        }
        outs = self.sess.run(None, feed)
        # Match outs to names
        # We’ll find by index matching to out_map order
        names = [o.name for o in self.sess.get_outputs()]
        # Build name->value map
        name2val = {names[i]: outs[i] for i in range(len(names))}
        prob = float(np.squeeze(name2val[self.out_map["prob"].name]))
        self.h = name2val[self.out_map["hn"].name]
        self.c = name2val[self.out_map["cn"].name]
        return prob

# ------------- Resampler & device helpers -------------

def linear_resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Simple linear resampler. x shape: (N, C)."""
    if src_sr == dst_sr or x.size == 0:
        return x.astype(np.float32, copy=False)
    n_src = x.shape[0]
    n_dst = int(round(n_src * (dst_sr / float(src_sr))))
    if n_dst <= 1 or n_src <= 1:
        return np.zeros((max(n_dst, 1), x.shape[1]), dtype=np.float32)
    t_src = np.linspace(0.0, 1.0, n_src, endpoint=False)
    t_dst = np.linspace(0.0, 1.0, n_dst, endpoint=False)
    y = np.empty((n_dst, x.shape[1]), dtype=np.float32)
    for c in range(x.shape[1]):
        y[:, c] = np.interp(t_dst, t_src, x[:, c]).astype(np.float32)
    return y

def choose_loop_device(loop_ids):
    """Pick the first loopback index that works; prefer stereo if available."""
    last_err = None
    for lid in loop_ids:
        try:
            dev = sd.query_devices(lid)
            loop_channels = 1 if dev["max_input_channels"] <= 1 else min(dev["max_input_channels"], 2)
            loop_rate = nearest_working_samplerate(lid, PREFERRED_OPEN_RATE, channels=loop_channels)
            return lid, dev, loop_channels, loop_rate
        except Exception as e:
            last_err = e
            print(f"Loopback device #{lid} not suitable: {e}")
            continue
    raise RuntimeError(f"No loopback device usable from {loop_ids}. Last error: {last_err}")

# ----------------- MAIN -----------------

def main():
    print("🛡️ Conversation Mode – sample-rate safe + stateful VAD (mic=2, loop=3→4 fallback)")

    # Mic
    mic_dev = sd.query_devices(MIC_ID)
    mic_rate = nearest_working_samplerate(MIC_ID, PREFERRED_OPEN_RATE, channels=1)
    print(f"Mic    #{MIC_ID}: '{mic_dev['name']}', max_in={mic_dev['max_input_channels']}, open_rate={mic_rate}")

    # Loopback (try 3 then 4)
    loop_id, loop_dev, loop_channels, loop_rate = choose_loop_device(LOOP_IDS)
    print(f"Loop   #{loop_id}: '{loop_dev['name']}', max_in={loop_dev['max_input_channels']}, "
          f"using_ch={loop_channels}, open_rate={loop_rate}")

    # Mixer / Ducking
    mixer = init_mixer()
    set_volume(mixer, NORMAL_VOLUME)

    # ONNX VAD (STATEFUL)
    vad = SileroVADStateful(ONNX_PATH, VAD_SAMPLE_RATE)

    # Frame math: gather enough native samples to make one 16k VAD frame
    vad_hop_16k = VAD_FRAME_16K
    mic_hop_native = int(round(vad_hop_16k * (mic_rate / float(VAD_SAMPLE_RATE))))
    loop_hop_native = int(round(vad_hop_16k * (loop_rate / float(VAD_SAMPLE_RATE))))
    aec_delay_frames = max(1, int(AEC_DELAY_FRAMES_16K))

    print(f"Frame config: VAD hop={vad_hop_16k} @16k | mic_hop={mic_hop_native} @ {mic_rate} | "
          f"loop_hop={loop_hop_native} @ {loop_rate} | AEC delay={aec_delay_frames} frames @16k")

    # History of loopback audio at 16k for AEC delay
    loop_history_16k = deque(maxlen=max(aec_delay_frames + 8, 16))

    ducked = False
    last_speech_time = 0.0

    # Accumulators (native domain) to make frame boundaries robust
    mic_accum = np.zeros((0, 1), dtype=np.float32)
    loop_accum = np.zeros((0, 1), dtype=np.float32)

    try:
        with sd.InputStream(device=MIC_ID, channels=1, samplerate=mic_rate, blocksize=mic_hop_native) as mic_in, \
             sd.InputStream(device=loop_id, channels=loop_channels, samplerate=loop_rate, blocksize=loop_hop_native) as loop_in:

            print("🚀 Streams opened successfully.")

            while True:
                # Read native chunks
                mic_chunk, _ = mic_in.read(mic_hop_native)                 # (M, 1)
                loop_chunk_multi, _ = loop_in.read(loop_hop_native)        # (L, C)
                loop_chunk = loop_chunk_multi[:, 0:1]                       # use channel 0

                # Accumulate
                mic_accum = np.vstack((mic_accum, mic_chunk.astype(np.float32)))
                loop_accum = np.vstack((loop_accum, loop_chunk.astype(np.float32)))

                # Resample accumulators to 16k
                mic_accum_16k = linear_resample(mic_accum, mic_rate, VAD_SAMPLE_RATE)
                loop_accum_16k = linear_resample(loop_accum, loop_rate, VAD_SAMPLE_RATE)

                # If not enough for one VAD hop, continue
                if mic_accum_16k.shape[0] < vad_hop_16k or loop_accum_16k.shape[0] < vad_hop_16k:
                    continue

                # Take one VAD frame at 16k
                mic_frame_16k = mic_accum_16k[:vad_hop_16k]
                loop_frame_16k = loop_accum_16k[:vad_hop_16k]

                # Consume approximately equivalent native samples
                consume_mic_native = int(round(vad_hop_16k * (mic_rate / float(VAD_SAMPLE_RATE))))
                consume_loop_native = int(round(vad_hop_16k * (loop_rate / float(VAD_SAMPLE_RATE))))
                mic_accum = mic_accum[consume_mic_native:]
                loop_accum = loop_accum[consume_loop_native:]

                # --- Simple AEC (delayed subtraction) ---
                loop_history_16k.append(loop_frame_16k.copy())
                if len(loop_history_16k) > aec_delay_frames:
                    delayed = loop_history_16k[-(aec_delay_frames + 1)]
                else:
                    delayed = np.zeros_like(loop_frame_16k)

                clean_16k = mic_frame_16k - 0.7 * delayed
                clean_16k = np.clip(clean_16k, -1.0, 1.0)

                # --- VAD (STATEFUL) ---
                prob = vad.forward(clean_16k)

                # --- Ducking logic ---
                now = time.time()
                if prob > VAD_THRESHOLD:
                    if not ducked:
                        print(f"🎤 Speech ({int(prob*100)}%) → Ducking music")
                        set_volume(mixer, DUCK_VOLUME)
                        ducked = True
                    last_speech_time = now
                elif ducked and (now - last_speech_time > UNDUCK_AFTER_SEC):
                    print("🔇 Silence → Restoring music")
                    set_volume(mixer, NORMAL_VOLUME)
                    ducked = False

    except KeyboardInterrupt:
        print("\n👋 Exiting by user.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Tips:")
        print("  • Your model seems stateful (x/h/c/sr). This script now feeds and rolls states each frame.")
        print("  • If you see 'Invalid number of channels', your loopback may be mono; the code already limits to 1–2 ch.")
        print("  • If streams fail on loopback #3, the script tries #4 automatically.")
        print("  • Ensure 'silero_vad.onnx' is the same stateful model you pulled.")
        raise

if __name__ == "__main__":
    main()