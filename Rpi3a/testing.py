import sounddevice as sd
import numpy as np

MIC_ID = 4
RATE = 48000
BLOCK = 1024

print("Opening mic silently…")
with sd.InputStream(device=MIC_ID, channels=1, samplerate=RATE, blocksize=BLOCK) as s:
    for i in range(20):
        data, _ = s.read(BLOCK)
        rms = float(np.sqrt(np.mean(data[:,0]**2)))
        print("RMS:", rms)