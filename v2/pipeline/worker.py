from typing import Optional
import numpy as np

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
        
        # --- Energy-based gating (suppress VAD on music/lyrics) ---
        mic_rms = float(np.sqrt(np.mean(cleaned**2) + 1e-9))
        ref_rms = float(np.sqrt(np.mean(ref_frame**2) + 1e-9)) if ref_frame is not None else 0.0
        
        # Gate 1: Absolute floor — if mic is very quiet, never call VAD
        too_quiet = mic_rms < 0.01
        
        # Gate 2: RMS gate — suppress VAD only when reference STRONGLY dominates
        ref_dominates = (ref_rms > 0.0 and mic_rms < 0.5 * ref_rms)
        
        # Gate 3: Coherence gate — suppress VAD when cleaned signal is echo-like
        echo_like = False
        if ref_frame is not None and len(ref_frame) == len(cleaned):
            mic_norm = cleaned / (np.linalg.norm(cleaned) + 1e-9)
            ref_norm = ref_frame / (np.linalg.norm(ref_frame) + 1e-9)
            coherence = float(np.dot(mic_norm, ref_norm))
            echo_like = coherence > 0.25
        
        # Step 4: Resample for VAD (48kHz -> 16kHz)
        vad_audio = self._resample_linear(cleaned, config.DEVICE_RATE, config.SAMPLE_RATE)
        
        # Step 5: Accumulate samples in buffer until we have enough for VAD
        self.vad_buffer = np.concatenate([self.vad_buffer, vad_audio])
        
        # Process VAD only when we have enough samples
        if len(self.vad_buffer) >= self.min_vad_samples:
            vad_chunk = self.vad_buffer[:self.min_vad_samples]
            self.vad_buffer = self.vad_buffer[self.min_vad_samples:]
            
            # Run VAD — only skip if mic is too quiet or ref strongly dominates
            raw_prob = 0.0
            if not too_quiet and not ref_dominates and not echo_like:
                speech_active = self.vad.update(vad_chunk)
                raw_prob = self.vad.get_probability()
            else:
                speech_active = False
            
            # Smoothing (matches working script behavior)
            self.last_vad_prob = (0.6 * self.last_vad_prob) + (0.4 * raw_prob)
            
            # Gate 4: Hard clamp — only if ref is active AND VAD is borderline
            # High confidence speech (>0.7) passes through even with ref active
            if ref_rms > 0.01 and self.last_vad_prob > 0.3 and self.last_vad_prob < 0.7:
                self.last_vad_prob = 0.0
                speech_active = False
            
            # Determine final speech state from smoothed probability
            speech_active = self.last_vad_prob >= 0.8
            self.last_speech_active = speech_active
            
            # Only duck if mic is actually louder than reference (near-field speech)
            if speech_active and mic_rms >= 0.02 and mic_rms > ref_rms:
                self.duck.notify_speech()
        
        # Always update ducking state (even when VAD not called)
        self.duck.update()
        
        self.frames_processed += 1
