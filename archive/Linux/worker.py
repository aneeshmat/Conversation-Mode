import threading
import time
import numpy as np

from aec_wrapper import AECWrapper
from audio_io import FRAME_SIZE
from vad_wrapper import setup_vad, warmup_vad, vad_prob_16k, SAMPLE_RATE
from classifier_inference import classifier_prob

DEVICE_RATE = 48000
VAD_ATTACK = 0.80
VAD_SMOOTHING = 0.6
HP_ALPHA = 0.995
GAIN_AFTER_AEC = 2.0

class DuckController:
    def __init__(self, ratio: float, hold_silence_sec: float,
                 get_volume_cb, set_volume_cb):
        self.ratio = float(ratio)
        self.hold = float(hold_silence_sec)
        self.ducked = False
        self.baseline = None
        self.target = None
        self.last_voice_ts = 0.0
        self.get_volume = get_volume_cb
        self.set_volume = set_volume_cb

    def _smooth_set_volume(self, from_p, to_p, steps=6, step_ms=30):
        import numpy as np, time
        steps = max(1, int(steps))
        if steps == 1 or from_p == to_p:
            self.set_volume(int(round(to_p)))
            return
        for t in np.linspace(from_p, to_p, steps):
            self.set_volume(int(round(t)))
            time.sleep(step_ms / 1000.0)

    def notify_speech(self):
        self.last_voice_ts = time.monotonic()
        if not self.ducked:
            base = self.get_volume()
            if base < 0:
                return
            tgt = int(round(base * self.ratio))
            tgt = max(0, min(150, tgt))
            self._smooth_set_volume(base, tgt)
            self.baseline = base
            self.target = tgt
            self.ducked = True

    def update(self):
        if not self.ducked:
            return
        now = time.monotonic()
        if (now - self.last_voice_ts) >= self.hold:
            current = self.get_volume()
            if current >= 0 and abs(current - self.target) <= 2:
                self._smooth_set_volume(current, self.baseline)
            self.ducked = False
            self.baseline = None
            self.target = None

    def stop(self):
        if not self.ducked:
            return
        current = self.get_volume()
        if (self.baseline is not None and current >= 0 and
            abs(current - self.target) <= 2):
            self._smooth_set_volume(current, self.baseline)
        self.ducked = False
        self.baseline = None
        self.target = None

def highpass_dc_block(x: np.ndarray, alpha: float = HP_ALPHA) -> np.ndarray:
    if x.size == 0:
        return x
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    for n in range(1, x.size):
        y[n] = x[n] - x[n-1] + alpha * y[n-1]
    return y

def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    src_n = len(x)
    if src_n == 0:
        return np.zeros(0, dtype=np.float32)
    dst_n = int(round(src_n * (dst_sr / src_sr)))
    if dst_n <= 1:
        return np.zeros(dst_n, dtype=np.float32)
    src_idx = np.linspace(0, src_n - 1, num=dst_n, dtype=np.float32)
    left = np.floor(src_idx).astype(np.int64)
    right = np.minimum(left + 1, src_n - 1)
    frac = src_idx - left
    y = (1.0 - frac) * x[left] + frac * x[right]
    return y.astype(np.float32)

class ConversationWorker:
    def __init__(self, audio_io, get_volume_cb, set_volume_cb,
                 duck_ratio=0.5, duck_hold=1.3):
        self.audio_io = audio_io
        self.aec = AECWrapper()
        self.duck = DuckController(duck_ratio, duck_hold,
                                   get_volume_cb, set_volume_cb)
        self.running = False
        self.thread = None
        self.last_vad_prob = 0.0
        self.smoothed_vad = 0.0

    def start(self):
        setup_vad()
        warmup_vad()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.duck.stop()
        self.aec.close()

    def _loop(self):
        q = self.audio_io.get_queue()
        while self.running:
            try:
                mic_dev, ref_dev = q.get(timeout=0.5)
            except Exception:
                self.duck.update()
                continue

            cleaned_dev = self.aec.process(ref_dev, mic_dev)
            cleaned_dev = highpass_dc_block(cleaned_dev, alpha=HP_ALPHA)

            mic_rms = float(np.sqrt(np.mean(cleaned_dev**2) + 1e-9))
            ref_rms = float(np.sqrt(np.mean(ref_dev**2) + 1e-9))

            cleaned_16k = resample_linear(cleaned_dev,
                                          src_sr=DEVICE_RATE,
                                          dst_sr=SAMPLE_RATE)
            if GAIN_AFTER_AEC != 1.0:
                cleaned_16k = cleaned_16k * np.float32(GAIN_AFTER_AEC)

            p_cls = classifier_prob(cleaned_dev, ref_dev)

            raw_prob = 0.0
            if cleaned_16k.size >= 512 and p_cls >= 0.8:
                raw_prob = vad_prob_16k(cleaned_16k[-1024:])

            self.smoothed_vad = (VAD_SMOOTHING * self.smoothed_vad) + \
                                ((1.0 - VAD_SMOOTHING) * raw_prob)
            self.last_vad_prob = float(max(0.0, min(1.0, self.smoothed_vad)))

            if ref_rms > 0.01 and self.last_vad_prob > 0.7 and p_cls < 0.9:
                self.last_vad_prob = 0.0

            if self.last_vad_prob >= VAD_ATTACK and mic_rms >= 0.02 and mic_rms > ref_rms:
                self.duck.notify_speech()

            self.duck.update()

    def get_vad_prob(self):
        return self.last_vad_prob
