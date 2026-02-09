# Conversation Mode v6: Lyric Shield for Raspberry Pi 3A+
import torch
import numpy as np
import sounddevice as sd
import threading
import time

# --- CONFIGURATION ---
SAMPLE_RATE = 16000
FRAME_SIZE = 512
# Adjust these based on 'python -m sounddevice'
MIC_ID = 1      # Your Microphone ID
LOOP_ID = 2     # The 'Loopback' Device ID

# Delay in frames (Sound travel time + Latency)
# 1 frame = ~32ms. If lyrics still trigger, try 1, 2, or 3.
DELAY_FRAMES = 2 

# Load Silero VAD (Use CPU mode for Pi 3A+)
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

# --- PROCESSING STATE ---
ring_buffer = np.zeros((FRAME_SIZE * (DELAY_FRAMES + 1), 1), dtype=np.float32)
last_prob = 0.0

def audio_callback(indata, outdata, frames, time_info, status):
    """
    This duplex stream reads from Mic (indata) and 
    we will simulate the subtraction here.
    Note: For true loopback, we'd pull from two streams, but 
    sounddevice allows multi-device streams if configured.
    """
    pass

def aec_loop():
    global last_prob, ring_buffer
    
    # We open the mic and the loopback simultaneously
    with sd.InputStream(device=MIC_ID, channels=1, samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE) as mic_in, \
         sd.InputStream(device=LOOP_ID, channels=1, samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE) as loop_in:
        
        print("🚀 Lyric Shield Active on RPi3A")
        
        while True:
            mic_chunk, _ = mic_in.read(FRAME_SIZE)
            loop_chunk, _ = loop_in.read(FRAME_SIZE)
            
            # 1. Manage Ring Buffer for Alignment
            # Shift buffer and add new loopback data
            ring_buffer = np.roll(ring_buffer, -FRAME_SIZE, axis=0)
            ring_buffer[-FRAME_SIZE:] = loop_chunk
            
            # 2. Get the 'Delayed' music (the music that would be hitting the mic now)
            delayed_music = ring_buffer[:FRAME_SIZE]
            
            # 3. SUBTRACTION (The Shield)
            # Music leakage into mic is usually quieter than the digital source.
            # 0.6 is the 'Leakage Factor'. Adjust 0.1 to 1.0 based on speaker loudness.
            clean_audio = mic_chunk - (delayed_music * 0.6)
            
            # 4. Normalize and VAD
            clean_audio = np.clip(clean_audio, -1.0, 1.0)
            audio_tensor = torch.from_numpy(clean_audio.astype(np.float32)).view(1, -1)
            
            # Silero VAD
            with torch.no_grad():
                prob = model(audio_tensor, SAMPLE_RATE).item()
            
            last_prob = prob
            
            # 5. Trigger Ducking Logic
            if prob > 0.65:
                print(f"🎤 Voice Detected ({int(prob*100)}%) - Ducking!")
                # Call your existing duck_volume() function here
            else:
                if prob > 0.3:
                    print(f"🎵 Filtered Music Leakage ({int(prob*100)}%)")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        aec_thread = threading.Thread(target=aec_loop, daemon=True)
        aec_thread.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")