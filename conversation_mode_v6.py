import onnxruntime as ort
import numpy as np
import sounddevice as sd
import alsaaudio
import time

# --- CONFIGURATION ---
MIC_ID = 1          # Your Microphone
LOOP_ID = 2         # The 'Loopback' device (snd-aloop)
SAMPLE_RATE = 16000
FRAME_SIZE = 512    # 32ms frames
DUCK_VOLUME = 20    # Volume level when you speak (%)
NORMAL_VOLUME = 80  # Volume level for music (%)

# --- INITIALIZE ALSA MIXER ---
# Most Pi setups use 'Master' or 'PCM'. Check with 'amixer scontrols'
try:
    mixer = alsaaudio.Mixer('PCM') 
except:
    mixer = alsaaudio.Mixer('Master')

# --- ONNX VAD SETUP ---
session = ort.InferenceSession("silero_vad.onnx")

# --- LYRIC SHIELD STATE ---
# Stores 4 frames of music for temporal alignment
ring_buffer = np.zeros((FRAME_SIZE * 4, 1), dtype=np.float32)
ducked = False

def set_pi_volume(val):
    mixer.setvolume(val)

def get_vad_prob(audio_frame):
    onnx_input = {
        "input": audio_frame.reshape(1, -1).astype(np.float32),
        "sr": np.array([SAMPLE_RATE], dtype=np.int64)
    }
    return session.run(None, onnx_input)[0][0][0]

def main():
    global ring_buffer, ducked
    print(f"🛡️ Conversation Mode V7 Active (RPi3A)")
    set_pi_volume(NORMAL_VOLUME)

    try:
        with sd.InputStream(device=MIC_ID, channels=1, samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE) as mic_in, \
             sd.InputStream(device=LOOP_ID, channels=1, samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE) as loop_in:
            
            while True:
                mic_chunk, _ = mic_in.read(FRAME_SIZE)
                loop_chunk, _ = loop_in.read(FRAME_SIZE)

                # 1. LYRIC SHIELD: Subtract music from mic
                ring_buffer = np.roll(ring_buffer, -FRAME_SIZE, axis=0)
                ring_buffer[-FRAME_SIZE:] = loop_chunk
                
                # Align music (1 frame ago) and subtract with 0.7 leakage factor
                delayed_music = ring_buffer[FRAME_SIZE:FRAME_SIZE*2]
                clean_audio = mic_chunk - (delayed_music * 0.7)
                clean_audio = np.clip(clean_audio, -1.0, 1.0)

                # 2. VAD: Check for human speech
                prob = get_vad_prob(clean_audio)

                # 3. DUCKING LOGIC
                if prob > 0.65:
                    if not ducked:
                        print(f"🎤 Speech detected ({int(prob*100)}%) - Ducking...")
                        set_pi_volume(DUCK_VOLUME)
                        ducked = True
                    last_speech_time = time.time()
                
                # Reset volume after 2 seconds of silence
                if ducked and (time.time() - last_speech_time > 2.0):
                    print("🔇 Silence detected - Restoring music...")
                    set_pi_volume(NORMAL_VOLUME)
                    ducked = False

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()