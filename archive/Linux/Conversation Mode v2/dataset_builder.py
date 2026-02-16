import os
import torch
import numpy as np
import soundfile as sf
from feature_extractor import extract_features

FRAME_SIZE_48K = 3072  # 64 ms @ 48k

# Your training WAV files
FILES = {
    "my_voice.wav": 1,
    "voice_over_music.wav": 1,
    "music_only.wav": 0,
    "silence.wav": 0,
}

def slice_frames(wav, frame_size):
    frames = []
    for i in range(0, len(wav) - frame_size, frame_size):
        frames.append(wav[i:i+frame_size])
    return frames

def main():
    X = []
    y = []

    for fname, label in FILES.items():
        if not os.path.exists(fname):
            print(f"Missing file: {fname}")
            continue

        print(f"Processing {fname}...")
        wav, sr = sf.read(fname)
        if wav.ndim > 1:
            wav = wav[:, 0]

        if sr != 48000:
            print(f"ERROR: {fname} must be 48 kHz, got {sr}")
            continue

        frames = slice_frames(wav, FRAME_SIZE_48K)

        for frame in frames:
            # Fake reference for training (we don't need real echo here)
            ref = np.zeros_like(frame)
            feats = extract_features(frame, ref)
            X.append(feats)
            y.append([label])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)

    print("Saving dataset...")
    torch.save(X, "features.pt")
    torch.save(y, "labels.pt")
    print("Done. Dataset ready.")

if __name__ == "__main__":
    main()
