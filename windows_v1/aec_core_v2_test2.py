import ctypes
import numpy as np
import pyaudio
import sys

# 1. Load DLL and Map Functions
try:
    aec_lib = ctypes.CDLL("./aec_core_v2_1.dll")
    aec_lib.aec_process_buffer.argtypes = [
        ctypes.c_void_p, 
        np.ctypeslib.ndpointer(dtype=np.float32), 
        np.ctypeslib.ndpointer(dtype=np.float32), 
        np.ctypeslib.ndpointer(dtype=np.float32), 
        ctypes.c_int,                             
        ctypes.c_int                              
    ]
    aec_lib.aec_create.restype = ctypes.c_void_p
    aec_lib.aec_free.argtypes = [ctypes.c_void_p]
except Exception as e:
    print(f"Failed to load DLL: {e}")
    sys.exit(1)

# 2. Hardware Config
MIC_ID = 1      
SPEAKER_ID = 6  
RATE = 44100    
CHUNK = 1024    

# Start with 0. If using Bluetooth (Willen), you might eventually 
# need to increase this to match the Bluetooth lag.
DELAY_SAMPLES = 441 

p = pyaudio.PyAudio()
aec_state = aec_lib.aec_create()

print(f"--- AEC REAL-WORLD SIMULATION ---")
print(f"Targeting Mic: {MIC_ID}, Speaker: {SPEAKER_ID}")

stream = None

try:
    stream = p.open(format=pyaudio.paInt16, 
                    channels=1, 
                    rate=RATE, 
                    input=True, 
                    output=True,
                    input_device_index=MIC_ID, 
                    output_device_index=SPEAKER_ID,
                    frames_per_buffer=CHUNK)

    print("\n[*] STATUS: Playing Tone. AEC is attempting to 'mute' it from the mic.")
    print("[*] Speak/Whistle to see if the AEC detects you vs the beep.")

    t_offset = 0
    while True:
        # 1. Capture real audio from the mic
        raw_mic = stream.read(CHUNK, exception_on_overflow=False)
        mic_float = np.frombuffer(raw_mic, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 2. Generate Reference Tone (The 'AI voice' simulation)
        # We create a 440Hz sine wave (A4 note)
        times = (np.arange(CHUNK) + t_offset) / RATE
        tone_ref = 0.3 * np.sin(2 * np.pi * 440 * times).astype(np.float32)
        t_offset += CHUNK
        
        filtered_output = np.zeros(CHUNK, dtype=np.float32)

        # 3. Process: Cancel the Tone (Reference) from the Mic (Input)
        aec_lib.aec_process_buffer(aec_state, tone_ref, mic_float, filtered_output, CHUNK, DELAY_SAMPLES)

        # 4. Clean up NaNs if any
        filtered_output = np.nan_to_num(filtered_output)

        # 5. DIAGNOSTICS
        mic_max = np.max(np.abs(mic_float))
        aec_max = np.max(np.abs(filtered_output))
        # This shows how much of the signal is left after the AEC does its job
        print(f"Mic Input Max: {mic_max:.4f} | AEC Result Max: {aec_max:.4f}", end='\r')

        # 6. OUTPUT: Send the Tone to the Marshall Willen
        # We play the 'tone_ref' so it exists in the room for the mic to hear
        out_int16 = (np.clip(tone_ref, -1, 1) * 32767.0).astype(np.int16)
        stream.write(out_int16.tobytes())

except KeyboardInterrupt:
    print("\n\n[*] Stopping...")
finally:
    if stream:
        stream.stop_stream()
        stream.close()
    p.terminate()
    if aec_state:
        aec_lib.aec_free(aec_state)