import sys
from demucs.apply import apply_model
from demucs.pretrained import get_model
import torchaudio
import torch
import os

# Usage: python3 separate_wav.py input.wav
if len(sys.argv) < 2:
    print("Usage: python3 separate_wav.py <input.wav>")
    sys.exit(1)

input_path = sys.argv[1]
if not os.path.exists(input_path):
    print(f"File not found: {input_path}")
    sys.exit(1)

# Try lighter models in order, fallback to htdemucs
model_names = ['demucs_v2', 'demucs_v3', 'demucs_v4', 'htdemucs']
for name in model_names:
    try:
        model = get_model(name)
        print(f"Loaded model: {name}")
        break
    except Exception as e:
        print(f"Model {name} not available: {e}")
else:
    print("No Demucs models available. Exiting.")
    sys.exit(1)
model.eval()

# Load audio
wav, sr = torchaudio.load(input_path)
if wav.shape[0] > 2:
    wav = wav[:2]  # Only stereo


# Convert mono to stereo if needed
if wav.shape[0] == 1:
    wav = wav.repeat(2, 1)
# Add batch dimension for Demucs
wav = wav.unsqueeze(0)
# Run separation
with torch.no_grad():
    sources = apply_model(model, wav, sr)

# Save separated sources
out_dir = input_path + "_demucs"
os.makedirs(out_dir, exist_ok=True)
for name, source in sources.items():
    out_path = os.path.join(out_dir, f"{name}.wav")
    torchaudio.save(out_path, source, sr)
    print(f"Saved: {out_path}")

print("Separation complete.")
