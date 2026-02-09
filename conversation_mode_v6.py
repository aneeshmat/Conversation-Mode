#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conversation Mode with Silero VAD
- Mic index = 2
- Loopback indices = 3, then 4 (fallback)
- Opens devices at a supported native rate, resamples to 16 kHz for VAD
- CPU-only ONNX (no GPU/DRM noise)
- Auto-detect loopback channels (1 or 2), picks channel 0
- Simple AEC (delayed subtraction) + volume ducking
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
    # desired first
    if desired: candidates.append(int(desired))
    # device default samplerate if present
    default_rate = int(dev.get("default_samplerate") or 0)
    if default_rate: candidates.append(default_rate)
    # common fallbacks
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
    # 1) ALSA
    if mixer is not None:
        try:
            mixer.setvolume(int(vol_percent))
            return
        except Exception as e:
            print(f"ALSA mixer setvolume failed: {e}")

    # 2) PipeWire (wpctl expects linear scalar 0.0–1.0)
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

    # 3) PulseAudio (pactl can use percentages)
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

def build_vad_session(onnx_path: str):
    """Create ONNX Runtime session (CPU only) and map input names (audio, sr)."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    in_name = None
    sr_name = None
    for i in inputs:
        # sr is usually int64 scalar/1D
        if i.type.startswith("tensor(int"):
            sr_name = i.name
        else:
            in_name = i.name
    # Fallback names if needed
    in_name = in_name or "input"
    sr_name = sr_name or "sr"
    return sess, in_name, sr_name

def get_vad_prob(session, in_name, sr_name, audio_frame_16k: np.ndarray) -> float:
    """Run Silero VAD; audio_frame_16k is (N,1) float32 at 16k."""
    x = audio_frame_16k.reshape(1, -1).astype(np.float32)
    sr = np.array([VAD_SAMPLE_RATE], dtype=np.int64)
    out = session.run(None, {in_name: x, sr_name: sr})[0]
    return float(np.squeeze(out))

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
    print("🛡️ Conversation Mode – sample-rate safe build (mic=2, loop=3→4 fallback)")
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

    # ONNX VAD
    session, in_name, sr_name = build_vad_session(ONNX_PATH)
    print(f"VAD ONNX inputs: audio='{in_name}', sr='{sr_name}'  | provider=CPU")

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

                # --- VAD ---
                prob = get_vad_prob(session, in_name, sr_name, clean_16k)

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
        print("  • If you see 'Invalid number of channels', your loopback may be mono; code already limits to 1–2 ch.")
        print("  • If streams fail on loopback #3, the script will try #4 automatically.")
        print("  • Ensure 'silero_vad.onnx' is present and compatible (expects inputs: audio, sr).")

if __name__ == "__main__":
    main()