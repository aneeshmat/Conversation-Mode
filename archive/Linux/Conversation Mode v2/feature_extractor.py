import numpy as np
import librosa

TARGET_SR = 16000
MIN_LEN = 512


def pad_signal(x, min_len=MIN_LEN):
    """Pad the signal to at least min_len samples."""
    if len(x) < min_len:
        x = np.pad(x, (0, min_len - len(x)), mode="constant")
    return x


def extract_features(cleaned_frame_48k, ref_frame_48k=None, sr_48k=48000):
    """
    Extract MFCC + spectral features from the cleaned microphone frame.
    Output dimension = 18 (matches your trained model).
    """

    # -----------------------------
    # 1. Resample to 16 kHz
    # -----------------------------
    x16 = librosa.resample(
        cleaned_frame_48k.astype(np.float32),
        orig_sr=sr_48k,
        target_sr=TARGET_SR,
        res_type="kaiser_fast"
    )

    # -----------------------------
    # 2. Pad to avoid n_fft warnings
    # -----------------------------
    x16 = pad_signal(x16, MIN_LEN)

    # -----------------------------
    # 3. MFCCs (13)
    # -----------------------------
    mfcc = librosa.feature.mfcc(
        y=x16,
        sr=TARGET_SR,
        n_mfcc=13,
        n_fft=512,
        hop_length=256
    )
    mfcc_mean = mfcc.mean(axis=1)

    # -----------------------------
    # 4. Spectral centroid (1)
    # -----------------------------
    centroid = librosa.feature.spectral_centroid(
        y=x16,
        sr=TARGET_SR,
        n_fft=512,
        hop_length=256
    ).mean()

    # -----------------------------
    # 5. Spectral flatness (1)
    # -----------------------------
    flatness = librosa.feature.spectral_flatness(
        y=x16,
        n_fft=512,
        hop_length=256
    ).mean()

    # -----------------------------
    # 6. RMS energy (1)
    # -----------------------------
    rms = librosa.feature.rms(
        y=x16,
        frame_length=512,
        hop_length=256
    ).mean()

    # -----------------------------
    # 7. Energy ratio (1)
    # -----------------------------
    S = np.abs(librosa.stft(
        x16,
        n_fft=512,
        hop_length=256
    ))
    low_energy = S[:40].mean()
    high_energy = S[40:].mean()
    energy_ratio = low_energy / (high_energy + 1e-6)

    # -----------------------------
    # 8. Zero-crossing rate (1)
    # -----------------------------
    zcr = librosa.feature.zero_crossing_rate(
        x16,
        frame_length=512,
        hop_length=256
    ).mean()

    # -----------------------------
    # 9. Final feature vector (18 dims)
    # -----------------------------
    features = np.concatenate([
        mfcc_mean,                          # 13
        np.array([
            centroid,                       # 14
            flatness,                       # 15
            rms,                            # 16
            energy_ratio,                   # 17
            zcr                              # 18
        ], dtype=np.float32)
    ])

    return features.astype(np.float32)

