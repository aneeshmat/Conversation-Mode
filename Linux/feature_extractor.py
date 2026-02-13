import numpy as np
import librosa

def extract_features(cleaned_48k, ref_48k, sr=48000):
    # Downsample to 16k for MFCCs
    cleaned_16k = librosa.resample(cleaned_48k, orig_sr=sr, target_sr=16000)
    ref_16k = librosa.resample(ref_48k, orig_sr=sr, target_sr=16000)

    # MFCCs (13 coefficients)
    mfcc = librosa.feature.mfcc(
        y=cleaned_16k,
        sr=16000,
        n_mfcc=13,
        n_fft=512,
        hop_length=512
    ).mean(axis=1)

    # RMS
    rms = np.sqrt(np.mean(cleaned_48k**2) + 1e-9)

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=cleaned_16k, sr=16000
    ).mean()

    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(
        y=cleaned_16k
    ).mean()

    # Coherence (normalized dot product)
    mic_norm = cleaned_48k / (np.linalg.norm(cleaned_48k) + 1e-9)
    ref_norm = ref_48k / (np.linalg.norm(ref_48k) + 1e-9)
    coherence = float(np.dot(mic_norm, ref_norm))

    # Energy ratio
    ref_rms = np.sqrt(np.mean(ref_48k**2) + 1e-9)
    ratio = rms / (ref_rms + 1e-9)

    return np.concatenate([
        mfcc,
        [rms, centroid, flatness, coherence, ratio]
    ]).astype(np.float32)

