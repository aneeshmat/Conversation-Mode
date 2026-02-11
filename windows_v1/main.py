import ctypes
import numpy as np
import sounddevice as sd
import os
import queue
import sys

# --- Load Engine ---
dll_path = os.path.abspath("aec_engine.dll")
aec_lib = ctypes.CDLL(dll_path)
aec_lib.aec_create.restype = ctypes.c_void_p
aec_lib.aec_process.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
aec_lib.aec_process.restype = ctypes.c_float
aec_lib.aec_free.argtypes = [ctypes.c_void_p]

# Hardware IDs - Ensure these are still correct
MIC_ID, LOOP_ID, SPEAKER_ID = 30, 29, 24   

def get_rate(device_id):
    return int(sd.query_devices(device_id)['default_samplerate'])

MIC_RATE, LOOP_RATE, OUT_RATE = get_rate(MIC_ID), get_rate(LOOP_ID), get_rate(SPEAKER_ID)

# Mixer Parameters
# vol_level is 0.4 now. Start low and slowly increase once confirmed stable.
vol_level = 0.4          
DUCK_THRESHOLD = 0.05    
ducking_multiplier = 1.0

state_ptr = aec_lib.aec_create()
mic_queue = queue.Queue(maxsize=10) # Ultra-small queue for lowest latency
ref_queue = queue.Queue(maxsize=10)

def mic_callback(indata, frames, time, status):
    try: mic_queue.put_nowait(indata.copy())
    except queue.Full: pass

def loopback_callback(indata, frames, time, status):
    try: ref_queue.put_nowait(indata[:, 0:1].copy())
    except queue.Full: pass

def output_callback(outdata, frames, time, status):
    global ducking_multiplier, vol_level
    try:
        m_block = mic_queue.get_nowait()
        r_block = ref_queue.get_nowait()

        # Voice detection (Ducking)
        mic_rms = np.sqrt(np.mean(m_block**2))
        if mic_rms > DUCK_THRESHOLD:
            ducking_multiplier = 0.1
        else:
            ducking_multiplier = min(1.0, ducking_multiplier + 0.01)

        count = min(len(m_block), len(r_block), frames)
        for i in range(count):
            # Process via C
            clean_s = aec_lib.aec_process(state_ptr, float(r_block[i, 0]), float(m_block[i, 0]))
            
            # Reduce gain of the cleaned signal slightly for stability
            outdata[i, 0] = clean_s * (vol_level * ducking_multiplier * 0.9)
            
        if count < frames: outdata[count:, 0] = 0
    except queue.Empty:
        outdata.fill(0)

print(f"--- AEC Priority Engine (Stabilized) ---\nStarting at 40% volume. Speak to test ducking.")

try:
    with sd.InputStream(device=MIC_ID, channels=1, samplerate=MIC_RATE, callback=mic_callback):
        with sd.InputStream(device=LOOP_ID, channels=2, samplerate=LOOP_RATE, callback=loopback_callback):
            with sd.OutputStream(device=SPEAKER_ID, channels=1, samplerate=OUT_RATE, callback=output_callback):
                while True: sd.sleep(100)
except Exception as e: print(f"Error: {e}")
finally: aec_lib.aec_free(state_ptr)