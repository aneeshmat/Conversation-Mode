import ctypes
import numpy as np
import pyaudio
import sys
import time

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

# --- DELAY CALIBRATION ---
# If the AEC isn't dropping the volume enough, increase this in steps of 500.
# For Bluetooth speakers, this can be as high as 4410 (100ms).
DELAY_SAMPLES = 0 

p = pyaudio.PyAudio()
aec_state = aec_lib.aec_create()

print(f"--- AEC REAL-WORLD SIMULATION V2 ---")
print(f"Using Delay: {DELAY_SAMPLES} samples")

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

    print("\n[*] STATUS: Playing Tone. AEC is 'muting' the speaker from the mic.")
    print("[*] Speak/Whistle to see the AEC distinguish YOU from the SPEAKER.")

    t_offset = 0
    while True:
        # 1. Capture real audio from the room
        raw_mic = stream.read(CHUNK, exception_on_overflow=False)
        mic_float = np.frombuffer(raw_mic, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 2. Generate Reference (Simulating AI speech with a 440Hz Tone)
        times = (np.arange(CHUNK) + t_offset) / RATE
        tone_ref = 0.3 * np.sin(2 * np.pi * 440 * times).astype(np.float32)
        t_offset += CHUNK
        
        filtered_output = np.zeros(CHUNK, dtype=np.float32)

        # 3. Process: The AEC "Brain" works here
        aec_lib.aec_process_buffer(aec_state, tone_ref, mic_float, filtered_output, CHUNK, DELAY_SAMPLES)

        # 4. Diagnostics: Calculate volume levels
        mic_max = np.max(np.abs(mic_float))
        aec_max = np.max(np.abs(filtered_output))
        
        # Calculate reduction percentage
        reduction = 0
        if mic_max > 0.01:
            reduction = max(0, (1 - (aec_max / mic_max)) * 100)

        # Print a live dashboard
        sys.stdout.write(f"\rMic: {mic_max:.4f} | AEC: {aec_max:.4f} | Reduction: {reduction:.1f}%")
        sys.stdout.flush()

        # 5. Output: Send the tone to the speaker
        # IMPORTANT: We play 'tone_ref' so it's in the room, 
        # but the AI's logic would use 'filtered_output' to "listen" to you.
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