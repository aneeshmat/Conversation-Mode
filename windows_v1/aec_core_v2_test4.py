import ctypes
import numpy as np
import pyaudio
import sys

# 1. Load DLL
aec_lib = ctypes.CDLL("./aec_core_v2_2.dll")
aec_lib.aec_process_buffer.argtypes = [ctypes.c_void_p, np.ctypeslib.ndpointer(dtype=np.float32), 
                                       np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.float32), 
                                       ctypes.c_int, ctypes.c_int]
aec_lib.aec_create.restype = ctypes.c_void_p

# 2. Config
RATE = 44100
CHUNK = 1024
MIC_ID = 1      # Your Microphone
# On Windows, you need the ID for "Stereo Mix" or "What U Hear"
# If you don't have this, we have to stick to the file method or use WASAPI.
LOOPBACK_ID = 2 

p = pyaudio.PyAudio()
aec_state = aec_lib.aec_create()

try:
    # MIC STREAM
    mic_stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                        input=True, input_device_index=MIC_ID, frames_per_buffer=CHUNK)
    
    # LOOPBACK STREAM (The "Reference")
    ref_stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                        input=True, input_device_index=LOOPBACK_ID, frames_per_buffer=CHUNK)

    print("[*] LIVE TEST: Play music on Spotify/YouTube now.")
    
    while True:
        # Capture what is playing out the speakers
        ref_data = ref_stream.read(CHUNK, exception_on_overflow=False)
        ref_float = np.frombuffer(ref_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Capture what the mic hears
        mic_data = mic_stream.read(CHUNK, exception_on_overflow=False)
        mic_float = np.frombuffer(mic_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        filtered_output = np.zeros(CHUNK, dtype=np.float32)

        # AEC: Subtract system audio from mic audio
        aec_lib.aec_process_buffer(aec_state, ref_float, mic_float, filtered_output, CHUNK, 4410)

        # Metrics for your Auto-Ducker
        mic_max = np.max(np.abs(mic_float))
        aec_max = np.max(np.abs(np.nan_to_num(filtered_output)))
        
        sys.stdout.write(f"\rMic (Music+Room): {mic_max:.4f} | AEC (Cleaned): {aec_max:.4f}  ")
        sys.stdout.flush()

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    p.terminate()