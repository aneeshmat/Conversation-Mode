"""
Main processing pipeline worker.
Coordinates audio capture, AEC, VAD, and ducking.
"""

import threading
import time
import numpy as np
from typing import Optional
from ..audio import AudioCapture, VolumeController
from ..aec import AECProcessor, AECStatus
from ..vad import SileroVAD
from ..ducking import DuckController
from .. import config


class ConversationWorker:
    """
    Main worker thread that runs the audio processing pipeline.
    """
    
    def __init__(self, audio_capture: AudioCapture, volume_controller: VolumeController):
        """
        Initialize worker.
        
        Args:
            audio_capture: Audio capture instance
            volume_controller: Volume controller instance
        """
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
        
        # Statistics
        self.frames_processed = 0
        self.last_vad_prob = 0.0
        self.last_speech_active = False
    
    def _highpass_dc_block(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply DC-blocking high-pass filter.
        
        Args:
            audio: Input audio samples
            
        Returns:
            Filtered audio
        """
        if len(audio) == 0:
            return audio
        
        alpha = config.HP_ALPHA
        output = np.zeros_like(audio)
        
        output[0] = audio[0] - self.hp_prev_in + alpha * self.hp_prev_out
        
        for i in range(1, len(audio)):
            output[i] = audio[i] - audio[i-1] + alpha * output[i-1]
        
        self.hp_prev_in = audio[-1]
        self.hp_prev_out = output[-1]
        
        return output
    
    def _resample_linear(self, audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """
        Resample audio using linear interpolation.
        
        Args:
            audio: Input audio
            src_rate: Source sample rate
            dst_rate: Destination sample rate
            
        Returns:
            Resampled audio
        """
        if src_rate == dst_rate:
            return audio
        
        src_len = len(audio)
        if src_len == 0:
            return np.zeros(0, dtype=np.float32)
        
        dst_len = int(round(src_len * (dst_rate / src_rate)))
        if dst_len <= 1:
            return np.zeros(dst_len, dtype=np.float32)
        
        # Linear interpolation
        src_indices = np.linspace(0, src_len - 1, num=dst_len, dtype=np.float32)
        left_indices = np.floor(src_indices).astype(np.int64)
        right_indices = np.minimum(left_indices + 1, src_len - 1)
        frac = src_indices - left_indices
        
        resampled = (1.0 - frac) * audio[left_indices] + frac * audio[right_indices]
        return resampled.astype(np.float32)
    
    def _process_frame(self, mic_frame: np.ndarray, ref_frame: Optional[np.ndarray]):
        """
        Process one audio frame through the pipeline.
        
        Args:
            mic_frame: Microphone audio
            ref_frame: Reference audio (speaker output), or None
        """
        # Step 1: Apply AEC
        if ref_frame is not None and len(ref_frame) == len(mic_frame):
            cleaned = self.aec.process(ref_frame, mic_frame)
        else:
            cleaned = mic_frame.copy()
        
        # Step 2: High-pass DC-blocking filter
        cleaned = self._highpass_dc_block(cleaned)
        
        # Step 3: Apply gain boost
        cleaned = cleaned * config.GAIN_AFTER_AEC
        
        # Clip to prevent overflow
        cleaned = np.clip(cleaned, -1.0, 1.0)
        
        # Step 4: Resample for VAD (48kHz -> 16kHz)
        vad_audio = self._resample_linear(cleaned, config.DEVICE_RATE, config.SAMPLE_RATE)
        
        # Step 5: Run VAD
        speech_active = self.vad.update(vad_audio)
        self.last_vad_prob = self.vad.get_probability()
        self.last_speech_active = speech_active
        
        # Step 6: Update ducking
        if speech_active:
            self.duck.notify_speech()
        
        self.duck.update()
        
        self.frames_processed += 1
    
    def _worker_loop(self):
        """Main worker loop."""
        if config.DEBUG:
            print("[Worker] Starting processing loop")
        
        while not self.stop_event.is_set():
            try:
                # Get audio frames
                mic_frame, ref_frame = self.audio.get_frames(timeout=0.1)
                
                if mic_frame is not None:
                    self._process_frame(mic_frame, ref_frame)
                
            except Exception as e:
                if config.DEBUG:
                    print(f"[Worker] Error in processing loop: {e}")
                time.sleep(0.01)
        
        if config.DEBUG:
            print("[Worker] Stopped processing loop")
    
    def start(self) -> bool:
        """
        Start the worker thread.
        
        Returns:
            True if started successfully
        """
        if self.running:
            return True
        
        # Start audio capture
        if not self.audio.start():
            return False
        
        # Reset components
        self.aec.reset()
        self.vad.reset()
        self.duck.reset()
        
        # Reset filter state
        self.hp_prev_in = 0.0
        self.hp_prev_out = 0.0
        self.frames_processed = 0
        
        # Start worker thread
        self.running = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        
        if config.DEBUG:
            print("[Worker] Started")
        
        return True
    
    def stop(self):
        """Stop the worker thread."""
        if not self.running:
            return
        
        if config.DEBUG:
            print("[Worker] Stopping...")
        
        # Signal stop
        self.running = False
        self.stop_event.set()
        
        # Wait for thread
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        
        # Stop audio
        self.audio.stop()
        
        # Force restore volume
        self.duck.force_restore()
        
        if config.DEBUG:
            print("[Worker] Stopped")
    
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self.running
    
    def get_vad_probability(self) -> float:
        """Get last VAD probability."""
        return self.last_vad_prob
    
    def is_speech_active(self) -> bool:
        """Check if speech is currently active."""
        return self.last_speech_active
    
    def is_ducked(self) -> bool:
        """Check if ducking is active."""
        return self.duck.is_ducked()
    
    def get_baseline_volume(self) -> int:
        """Get baseline volume (-1 if not ducked)."""
        return self.duck.get_baseline()
    
    def get_aec_status(self) -> AECStatus:
        """Get AEC status."""
        return self.aec.get_status()
    
    def set_aec_enabled(self, enabled: bool):
        """Enable or disable AEC."""
        self.aec.set_enabled(enabled)
        if not enabled:
            self.aec.reset()
    
    def get_stats(self) -> dict:
        """Get processing statistics."""
        return {
            'frames_processed': self.frames_processed,
            'vad_probability': self.last_vad_prob,
            'speech_active': self.last_speech_active,
            'ducked': self.duck.is_ducked(),
            'baseline_volume': self.duck.get_baseline(),
            'aec_status': self.aec.get_status().value
        }
