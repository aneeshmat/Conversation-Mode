import onnxruntime as ort
import numpy as np
import sounddevice as sd
import alsaaudio
import time
import os

# Suppress ONNX GPU warnings
os.environ["ORT_LOGGING_LEVEL"] = "3"

# --- CONFIGURATION (Based on your hw:x,y output) ---
MIC_ID = 2      # hw:2,0 (USB PNP Audio)
LOOP_ID = 3     # hw:3,0 (Loopback PCM)
SAMPLE_RATE = 16000
FRAME_SIZE = 512
DUCK_VOLUME = 20
NORMAL_VOLUME = 80

# --- INITIALIZE ALSA MIXER ---
try:
    mixer = alsaaudio.Mixer('PCM') 
except:
    try:
        mixer = alsaaudio.Mixer('Master')
    except:
        mixer = None
        print("Warning: Could not find Mixer. Volume ducking may not work.")

# --- ONNX VAD SETUP ---
session = ort.InferenceSession("silero_vad.onnx")

# --- STATE ---
ring_buffer = np.zeros((FRAME_SIZE * 4, 1), dtype=np.float32)
ducked = False
last_speech_time = 0

def get_vad_prob(audio_frame):
    onnx_input = {
        "input": audio_frame.reshape(1, -1).astype(np.float32),
        "sr": np.array([SAMPLE_RATE], dtype=np.int64)
    }
    return session.run(None, onnx_input)[0][0][0]

def main():
    global ring_buffer, ducked, last_speech_time
    print(f"🛡️ Conversation Mode V7.2 (32-Channel Fix)")
    if mixer: mixer.setvolume(NORMAL_VOLUME)

    try:
        # OPEN STREAMS: 
        # Mic is 1 channel (from your output hw:2,0)
        # Loopback is 32 channels (from your output hw:3,0)
        with sd.InputStream(device=MIC_ID, channels=1, samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE) as mic_in, \
             sd.InputStream(device=LOOP_ID, channels=32, samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE) as loop_in:
            
            print("🚀 Streams opened successfully!")
            
            while True:
                mic_chunk, _ = mic_in.read(FRAME_SIZE)
                loop_chunk_multi, _ = loop_in.read(FRAME_SIZE)
                
                # Take only the first channel of the 32 loopback channels
                loop_chunk = loop_chunk_multi[:, 0:1] 

                # 1. LYRIC SHIELD (AEC Subtraction)
                ring_buffer = np.roll(ring_buffer, -FRAME_SIZE, axis=0)
                ring_buffer[-FRAME_SIZE:] = loop_chunk
                
                # Align music and subtract
                delayed_music = ring_buffer[FRAME_SIZE:FRAME_SIZE*2]
                clean_audio = mic_chunk - (delayed_music * 0.7)
                clean_audio = np.clip(clean_audio, -1.0, 1.0)

                # 2. VAD PROCESSING
                prob = get_vad_prob(clean_audio)

                # 3. DUCKING LOGIC
                if prob > 0.65:
                    if not ducked:
                        print(f"🎤 Speech ({int(prob*100)}%) - Ducking Music")
                        if mixer: mixer.setvolume(DUCK_VOLUME)
                        ducked = True
                    last_speech_time = time.time()
                
                if ducked and (time.time() - last_speech_time > 2.5):
                    print("🔇 Silence - Restoring Music")
                    if mixer: mixer.setvolume(NORMAL_VOLUME)
                    ducked = False

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Tip: If you see 'Invalid number of channels', check 'python3 -m sounddevice' again.")

if __name__ == "__main__":
    main()