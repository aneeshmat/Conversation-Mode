import numpy as np

SAMPLE_RATE = 16000
VAD_WINDOW_16K = 1024

USE_TORCH = True
model = None
vad_session = None

def setup_vad():
    global model, vad_session, USE_TORCH
    try:
        import torch
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        model.eval()
        USE_TORCH = True
        print("✅ Silero VAD (Torch) loaded")
    except Exception as e:
        print(f"⚠️ Torch/Silero VAD load failed: {e}\n→ Trying ONNX Runtime fallback...")
        try:
            import onnxruntime as ort
            import urllib.request
            import pathlib
            MODEL_PATH = pathlib.Path("silero_vad.onnx")
            if not MODEL_PATH.exists():
                print("Downloading Silero VAD ONNX model...")
                urllib.request.urlretrieve(
                    "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx",
                    MODEL_PATH
                )
            vad_session = ort.InferenceSession(str(MODEL_PATH),
                                               providers=["CPUExecutionProvider"])
            USE_TORCH = False
            print("✅ Silero VAD (ONNX) loaded")
        except Exception as e2:
            print(f"❌ ONNX fallback failed: {e2}")
            print("VAD unavailable. Speech detection will not work.")

def warmup_vad():
    dummy = np.zeros(512, dtype=np.float32)
    _ = vad_prob_16k(dummy)

def _torch_vad_prob_16k(audio_16k: np.ndarray) -> float:
    import torch
    n = audio_16k.size
    if n >= 1024:
        seg = audio_16k[-1024:]
        wins = [seg[:512], seg[512:]]
        probs = []
        with torch.no_grad():
            for w in wins:
                t = torch.from_numpy(w.astype(np.float32))
                probs.append(float(model(t, SAMPLE_RATE).item()))
        return max(probs)
    elif n >= 512:
        w = audio_16k[-512:]
        with torch.no_grad():
            t = torch.from_numpy(w.astype(np.float32))
            return float(model(t, SAMPLE_RATE).item())
    else:
        return 0.0

def vad_prob_16k(audio_16k: np.ndarray) -> float:
    if USE_TORCH and model is not None:
        return _torch_vad_prob_16k(audio_16k)
    elif (not USE_TORCH) and vad_session is not None:
        inp_name = vad_session.get_inputs()[0].name
        x = audio_16k.astype(np.float32)[None, :]
        out = vad_session.run(None, {inp_name: x})[0]
        return float(out.ravel()[0])
    else:
        return 0.0
