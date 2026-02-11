import ctypes
import numpy as np
import pyaudio
import sys

# 1. Load DLL
try:
    aec_lib = ctypes.CDLL("./aec_core_v2.dll")
    aec_lib.aec_process_buffer.argtypes = [
        ctypes.c_void_p, 
        np.ctypeslib.ndpointer(dtype=np.float32), 
        np.ctypeslib.ndpointer(dtype=np.float32), 
        np.ctypeslib.ndpointer(dtype=np.float32), 
        ctypes.c_int,                             
        ctypes.c_int                              
    ]
    aec_lib.aec_create.restype = ctypes.c_void_p
    # FIX: Explicitly define aec_free to handle 64-bit pointers correctly
    aec_lib.aec_free.argtypes = [ctypes.c_void_p]
    
except Exception as e:
    print(f"Failed to load DLL: {e}")
    sys.exit(1)

# 2. Hardware Config
MIC_ID = 1      
SPEAKER_ID = 6  
RATE = 44100    
CHUNK = 1024    
DELAY_SAMPLES = 500

p = pyaudio.PyAudio()
aec_state = aec_lib.aec_create()

print(f"--- AEC STABILITY TEST ---")
stream = None

try:
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, 
                    input=True, output=True,
                    input_device_index=MIC_ID, output_device_index=SPEAKER_ID,
                    frames_per_buffer=CHUNK)

    print("[*] TEST: Whistle steadily. It should start loud and then fade away.")

    while True:
        raw_mic = stream.read(CHUNK, exception_on_overflow=False)
        mic_float = np.frombuffer(raw_mic, dtype=np.int16).astype(np.float32) / 32768.0
        filtered_output = np.zeros(CHUNK, dtype=np.float32)

        aec_lib.aec_process_buffer(aec_state, mic_float, mic_float, filtered_output, CHUNK, DELAY_SAMPLES)
        print(f"Max Output: {np.max(np.abs(filtered_output)):.4f}", end='\r')
        #aec_lib.aec_process_buffer(aec_state, np.zeros(CHUNK, dtype=np.float32), mic_float, filtered_output, CHUNK, 0)
        # Safety check: replace NaNs with 0 to prevent crash
        filtered_output = np.nan_to_num(filtered_output)

        out_int16 = (np.clip(filtered_output, -1, 1) * 32767.0).astype(np.int16)
        stream.write(out_int16.tobytes())

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    if stream:
        stream.close()
    p.terminate()
    if aec_state:
        aec_lib.aec_free(aec_state)