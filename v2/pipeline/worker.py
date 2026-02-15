"""
Main processing pipeline worker.
Coordinates audio capture, AEC, VAD, and ducking.
"""

import threading
import time
import numpy as np
from typing import Optional

# Handle both package and direct execution
try:
    from ..audio import AudioCapture, VolumeController
    from ..aec import AECProcessor, AECStatus
    from ..vad import SileroVAD
    from ..ducking import DuckController
    from .. import config
except ImportError:
    from audio import AudioCapture, VolumeController
    from aec import AECProcessor, AECStatus
    from vad import SileroVAD
    from ducking import DuckController
    import config


class ConversationWorker:
    """
    Main worker thread that runs the audio processing pipeline.
    """

    def __init__(self, audio_capture: AudioCapture, volume_controller: VolumeController):
        self.audio = audio_capture
        self.volume = volume_controller

        # Initialize components
        self.aec = AECProcessor(
            frame_size=config.FRAME_SIZE,
            sample_rate=config.DEVICE_RATE,
            enabled=config.AEC_ENABLED_DEFAULT
        )

        self.vad = SileroVAD(sample_rate=config.SAMPLE_RATE)

        self.duck = DuckController(
            get_volume_fn=volume_controller.get_volume,
            set_volume_fn=volume_controller.set_volume
        )

        # Worker thread
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.stop_event = threading.Event()

        # DC-blocking filter state
        self.hp_prev_in = 0.0
        self.hp_prev_out = 0.0

        # VAD accumulation buffer
        self.vad_buffer = np.array([], dtype=np.float32)
        self.min_vad_samples = 512  # Silero VAD minimum

        # Statistics
        self.frames_processed = 0
        self.last_vad_prob = 0.0
        self.last_speech_active = False

    # -------------------------------------------------------------
    # High-pass DC-blocking filter
    # -------------------------------------------------------------
    def _highpass_dc_block(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio

        alpha = config.HP_ALPHA
        output = np.zeros_like(audio)

        output[0] = audio[0] - self.hp_prev_in + alpha * self.hp_prev_out

        for i in range(1, len(audio)):
            output[i] = audio[i] - audio[i - 1] + alpha * output[i - 1]

        self.hp_prev_in = audio[-1]
        self.hp_prev_out = output[-1]

        return output

    # -------------------------------------------------------------
    # Linear resampler
    # -------------------------------------------------------------
    def _resample_linear(self, audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate:
            return audio

        src_len = len(audio)
        if src_len == 0:
            return np.zeros(0, dtype=np.float32)

        dst_len = int(round(src_len * (dst_rate / src_rate)))
        if dst_len <= 1:
            return np.zeros(dst_len, dtype=np.float32)

        src_indices = np.linspace(0, src_len - 1, num=dst_len, dtype=np.float32)
        left = np.floor(src_indices).astype(np.int64)
        right = np.minimum(left + 1, src_len - 1)
        frac = src_indices - left

        resampled = (1.0 - frac) * audio[left] + frac * audio[right]
        return resampled.astype(np.float32)

    # -------------------------------------------------------------
    # Frame processor
    # -------------------------------------------------------------
    def _process_frame(self, mic_frame: np.ndarray, ref_frame: Optional[np.ndarray]):
        # Step 1: AEC
        if ref_frame is not None and len(ref_frame) == len(mic_frame):
            cleaned = self.aec.process(ref_frame, mic_frame)
        else:
            cleaned = mic_frame.copy()

        # Step 2: High-pass filter
        cleaned = self._highpass_dc_block(cleaned)

        # Step 3: Gain
        cleaned = cleaned * config.GAIN_AFTER_AEC
        cleaned = np.clip(cleaned, -1.0, 1.0)

        # Step 4: Energy gating
        mic_rms = float(np.sqrt(np.mean(cleaned**2) + 1e-9))
        ref_rms = float(np.sqrt(np.mean(ref_frame**2) + 1e-9)) if ref_frame is not None else 0.0

        too_quiet = mic_rms < 0.01
        ref_dominates = (ref_rms > 0.0 and mic_rms < 0.5 * ref_rms)

        echo_like = False
        if ref_frame is not None and len(ref_frame) == len(cleaned):
            mic_norm = cleaned / (np.linalg.norm(cleaned) + 1e-9)
            ref_norm = ref_frame / (np.linalg.norm(ref_frame) + 1e-9)
            coherence = float(np.dot(mic_norm, ref_norm))
            echo_like = coherence > 0.25

        # Step 5: Resample for VAD
        vad_audio = self._resample_linear(cleaned, config.DEVICE_RATE, config.SAMPLE_RATE)

        # Step 6: Accumulate VAD buffer
        self.vad_buffer = np.concatenate([self.vad_buffer, vad_audio])

        if len(self.vad_buffer) >= self.min_vad_samples:
            vad_chunk = self.vad_buffer[:self.min_vad_samples]
            self.vad_buffer = self.vad_buffer[self.min_vad_samples:]

            # Run VAD only if gates allow
            if not too_quiet and not ref_dominates and not echo_like:
                self.vad.update(vad_chunk)
                raw_prob = self.vad.get_probability()
                speech_active = raw_prob > 0.5
            else:
                raw_prob = 0.0
                speech_active = False

            # Hard clamp when reference is active
            if ref_rms > 0.01 and raw_prob > 0.3:
                raw_prob = 0.0
                speech_active = False

            self.last_vad_prob = raw_prob
            self.last_speech_active = speech_active

            # Step 7: Ducking
            if speech_active and mic_rms >= 0.02 and mic_rms > ref_rms:
                self.duck.notify_speech()

        # Always update ducking state
        self.duck.update()

        self.frames_processed += 1

    # -------------------------------------------------------------
    # Worker loop
    # -------------------------------------------------------------
    def _worker_loop(self):
        if config.DEBUG:
            print("[Worker] Starting processing loop")

        while not self.stop_event.is_set():
            try:
                mic_frame, ref_frame = self.audio.get_frames(timeout=0.1)
                if mic_frame is not None:
                    self._process_frame(mic_frame, ref_frame)

            except Exception as e:
                if config.DEBUG:
                    print(f"[Worker] Error in processing loop: {e}")
                time.sleep(0.01)

        if config.DEBUG:
            print("[Worker] Stopped processing loop")

    # -------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------
    def start(self) -> bool:
        if self.running:
            return True

        if not self.audio.start():
            return False

        self.aec.reset()
        self.vad.reset()
        self.duck.reset()

        self.hp_prev_in = 0.0
        self.hp_prev_out = 0.0
        self.vad_buffer = np.array([], dtype=np.float32)
        self.frames_processed = 0

        self.running = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

        if config.DEBUG:
            print("[Worker] Started")

        return True

    def stop(self):
        if not self.running:
            return

        if config.DEBUG:
            print("[Worker] Stopping...")

        self.running = False
        self.stop_event.set()

        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None

        self.audio.stop()
        self.duck.force_restore()

        if config.DEBUG:
            print("[Worker] Stopped")

    def is_running(self) -> bool:
        return self.running

    def get_vad_probability(self) -> float:
        return self.last_vad_prob

    def is_speech_active(self) -> bool:
        return self.last_speech_active

    def is_ducked(self) -> bool:
        return self.duck.is_ducked()

    def get_baseline_volume(self) -> int:
        return self.duck.get_baseline()

    def get_aec_status(self) -> AECStatus:
        return self.aec.get_status()

    def set_aec_enabled(self, enabled: bool):
        self.aec.set_enabled(enabled)
        if not enabled:
            self.aec.reset()

    def get_stats(self) -> dict:
        return {
            'frames_processed': self.frames_processed,
            'vad_probability': self.last_vad_prob,
            'speech_active': self.last_speech_active,
            'ducked': self.duck.is_ducked(),
            'baseline_volume': self.duck.get_baseline(),
            'aec_status': self.aec.get_status().value
        }
