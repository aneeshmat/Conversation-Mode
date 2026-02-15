"""
Silero VAD wrapper with PyTorch and ONNX Runtime fallback.
Includes EMA smoothing and hysteresis for stable speech detection.
"""

import numpy as np
import torch
from typing import Optional
from .. import config


class SileroVAD:
    """
    Silero VAD with EMA smoothing and hysteresis.
    Supports PyTorch with ONNX Runtime fallback.
    """
    
    def __init__(self, sample_rate: int = config.SAMPLE_RATE):
        """
        Initialize Silero VAD.
        
        Args:
            sample_rate: Audio sample rate (must be 8000 or 16000 Hz)
        """
        if sample_rate not in [8000, 16000]:
            raise ValueError(f"Silero VAD only supports 8kHz or 16kHz, got {sample_rate}Hz")
        
        self.sample_rate = sample_rate
        self.model = None
        self.using_torch = False
        
        # VAD state
        self.prob_ema = 0.0
        self.speech_active = False
        self.hold_counter = 0
        
        # Configuration
        self.on_threshold = config.VAD_ON_THRESHOLD
        self.off_threshold = config.VAD_OFF_THRESHOLD
        self.ema_alpha = config.VAD_EMA_ALPHA
        self.hold_frames = config.VAD_HOLD_FRAMES
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load Silero VAD model (PyTorch or ONNX)."""
        try:
            # Try PyTorch first
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            self.using_torch = True
            
            if config.DEBUG:
                print("[VAD] Loaded Silero VAD (PyTorch)")
                
        except Exception as e:
            if config.DEBUG:
                print(f"[VAD] PyTorch load failed ({e}), trying ONNX...")
            
            try:
                # Try ONNX Runtime fallback
                import onnxruntime
                self._load_onnx_model()
                
                if config.DEBUG:
                    print("[VAD] Loaded Silero VAD (ONNX)")
                    
            except Exception as e2:
                raise RuntimeError(f"Failed to load Silero VAD: PyTorch failed ({e}), ONNX failed ({e2})")
    
    def _load_onnx_model(self):
        """Load ONNX model (fallback)."""
        import onnxruntime
        import tempfile
        import os
        from pathlib import Path
        
        # Download ONNX model if needed
        model_path = Path.home() / ".cache" / "silero_vad" / "silero_vad.onnx"
        
        if not model_path.exists():
            # Use torch hub to download, then convert
            model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            
            # Create cache directory
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export to ONNX
            dummy_input = torch.randn(1, 512)
            torch.onnx.export(
                model,
                dummy_input,
                str(model_path),
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch', 1: 'sequence'}}
            )
        
        # Load ONNX model
        self.model = onnxruntime.InferenceSession(str(model_path))
        self.using_torch = False
    
    def process_frame(self, audio_frame: np.ndarray) -> float:
        """
        Process audio frame and return raw VAD probability.
        
        Args:
            audio_frame: Audio samples (float32, normalized to -1 to 1)
            
        Returns:
            Speech probability (0.0 to 1.0)
        """
        if self.model is None:
            return 0.0
        
        # Ensure correct shape and type
        if len(audio_frame.shape) == 1:
            audio_frame = audio_frame.reshape(1, -1)
        
        try:
            if self.using_torch:
                # PyTorch inference
                with torch.no_grad():
                    tensor = torch.from_numpy(audio_frame).float()
                    prob = self.model(tensor, self.sample_rate).item()
            else:
                # ONNX inference
                input_name = self.model.get_inputs()[0].name
                prob = self.model.run(None, {input_name: audio_frame.astype(np.float32)})[0][0]
            
            return float(prob)
            
        except Exception as e:
            if config.DEBUG:
                print(f"[VAD] Inference error: {e}")
            return 0.0
    
    def update(self, audio_frame: np.ndarray) -> bool:
        """
        Update VAD state with new audio frame.
        Applies EMA smoothing and hysteresis.
        
        Args:
            audio_frame: Audio samples (float32, normalized to -1 to 1)
            
        Returns:
            True if speech is active, False otherwise
        """
        # Get raw probability
        raw_prob = self.process_frame(audio_frame)
        
        # Apply EMA smoothing
        self.prob_ema = (self.ema_alpha * raw_prob + 
                         (1 - self.ema_alpha) * self.prob_ema)
        
        # Hysteresis logic
        if not self.speech_active:
            # Currently silent - check if speech starts
            if self.prob_ema >= self.on_threshold:
                self.speech_active = True
                self.hold_counter = self.hold_frames
        else:
            # Currently speaking
            if self.prob_ema >= self.off_threshold:
                # Still speaking - reset hold counter
                self.hold_counter = self.hold_frames
            else:
                # Probability dropped - decrement hold counter
                self.hold_counter -= 1
                if self.hold_counter <= 0:
                    self.speech_active = False
        
        return self.speech_active
    
    def get_probability(self) -> float:
        """Get current smoothed VAD probability."""
        return self.prob_ema
    
    def is_speech_active(self) -> bool:
        """Check if speech is currently active."""
        return self.speech_active
    
    def reset(self):
        """Reset VAD state."""
        self.prob_ema = 0.0
        self.speech_active = False
        self.hold_counter = 0
        
        # Reset model state if PyTorch
        if self.using_torch and self.model is not None:
            try:
                # Silero VAD has internal state that should be reset
                # Process a silent frame to reset
                silent = np.zeros(512, dtype=np.float32)
                self.process_frame(silent)
            except Exception:
                pass
    
    def set_thresholds(self, on_threshold: float, off_threshold: float):
        """
        Update VAD thresholds.
        
        Args:
            on_threshold: Probability threshold to trigger speech (0.0-1.0)
            off_threshold: Probability threshold to end speech (0.0-1.0)
        """
        self.on_threshold = max(0.0, min(1.0, on_threshold))
        self.off_threshold = max(0.0, min(1.0, off_threshold))
