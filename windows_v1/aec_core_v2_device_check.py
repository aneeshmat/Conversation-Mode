import pyaudio
import numpy as np
import sys

# --- CONFIGURATION ---
# Based on your previous output:
MIC_ID = 1        # Microphone (4- USB PnP Audio Device)
SPEAKER_ID = 6    # Speakers (WILLEN)
RATE = 44100      # 44.1kHz is safer for Bluetooth/USB combos than 48kHz
CHUNK = 1024      # Larger chunk size helps prevent the -9999 Host Error
FORMAT = pyaudio.paInt16 # 16-bit is required for many USB PnP Mics

p = pyaudio.PyAudio()

print("--- HARDWARE DIAGNOSTIC ---")
print(f"Targeting Mic ID: {MIC_ID}")
print(f"Targeting Speaker ID: {SPEAKER_ID}")

try:
    # Verify IDs exist
    mic_info = p.get_device_info_by_index(MIC_ID)
    spk_info = p.get_device_info_by_index(SPEAKER_ID)
    print(f"Using Mic: {mic_info['name']}")
    print(f"Using Speaker: {spk_info['name']}")
    
    stream = p.open(
        format=FORMAT,
        channels=1,
        rate=RATE,
        input=True,
        output=True,
        input_device_index=MIC_ID,
        output_device_index=SPEAKER_ID,
        frames_per_buffer=CHUNK
    )

    print("\n[*] STATUS: STREAM OPEN SUCCESSFUL")
    print("[*] Speak into the mic. You should hear yourself in the Willen speaker.")
    print("[*] Press Ctrl+C to stop the test.\n")

    while True:
        # Read data from mic
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # Calculate Volume for Visual Feedback
        samples = np.frombuffer(data, dtype=np.int16)
        volume = np.linalg.norm(samples) / np.sqrt(len(samples))
        
        # Simple text-based volume meter
        meter = "#" * int(volume / 100)
        sys.stdout.write(f"\rVolume: [{meter:<50}] {volume:.2f}  ")
        sys.stdout.flush()

        # Play data back through speakers
        stream.write(data)

except KeyboardInterrupt:
    print("\n\n[*] Stopping test...")

except Exception as e:
    print(f"\n[!] ERROR: {e}")
    print("\nTROUBLESHOOTING STEPS:")
    print("1. Go to Windows Sound Settings > More Sound Settings.")
    print("2. Ensure BOTH devices are set to 44100Hz in 'Advanced Properties'.")
    print("3. Ensure 'Allow applications to take exclusive control' is UNCHECKED if the error persists.")
    print("4. Try changing RATE to 48000 in this script if 44100 fails.")

finally:
    if 'stream' in locals():
        stream.stop_stream()
        stream.close()
    p.terminate()