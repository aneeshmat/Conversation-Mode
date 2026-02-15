"""
Pure Python fallback AEC using spectral subtraction.
Simpler than full adaptive filtering but doesn't require SpeexDSP.
"""

import numpy as np
from typing import Optional


class AECFallback:
    """
    Simple spectral subtraction-based echo cancellation fallback.
    Not as effective as SpeexDSP but provides basic echo reduction.
    """
    
    def __init__(self, frame_size: int, sample_rate: int):
        """
        Initialize fallback AEC.
        
        Args:
            frame_size: Samples per frame
            sample_rate: Audio sample rate in Hz
        """
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        
        # FFT size (next power of 2)
        self.fft_size = 1 << (frame_size - 1).bit_length()
        
        # Spectral smoothing
        self.ref_magnitude_smooth = None
        self.alpha = 0.7  # Smoothing factor
        
        # Over-subtraction factor
        self.beta = 2.0
        
        # Spectral floor to prevent over-suppression
        self.floor = 0.01
    
    def process(self, ref_frame: np.ndarray, mic_frame: np.ndarray) -> np.ndarray:
        """
        Process audio using spectral subtraction.
        
        Args:
            ref_frame: Reference signal (speaker output)
            mic_frame: Microphone signal
            
        Returns:
            Echo-reduced signal
        """
        # Pad to FFT size
        ref_padded = np.zeros(self.fft_size, dtype=np.float32)
        mic_padded = np.zeros(self.fft_size, dtype=np.float32)
        
        ref_padded[:len(ref_frame)] = ref_frame
        mic_padded[:len(mic_frame)] = mic_frame
        
        # Apply window (Hann)
        window = np.hanning(self.frame_size)
        ref_windowed = ref_padded.copy()
        mic_windowed = mic_padded.copy()
        ref_windowed[:self.frame_size] *= window
        mic_windowed[:self.frame_size] *= window
        
        # FFT
        ref_fft = np.fft.rfft(ref_windowed)
        mic_fft = np.fft.rfft(mic_windowed)
        
        # Magnitude and phase
        ref_mag = np.abs(ref_fft)
        mic_mag = np.abs(mic_fft)
        mic_phase = np.angle(mic_fft)
        
        # Smooth reference magnitude
        if self.ref_magnitude_smooth is None:
            self.ref_magnitude_smooth = ref_mag
        else:
            self.ref_magnitude_smooth = (self.alpha * self.ref_magnitude_smooth + 
                                         (1 - self.alpha) * ref_mag)
        
        # Spectral subtraction
        out_mag = mic_mag - self.beta * self.ref_magnitude_smooth
        
        # Apply floor
        out_mag = np.maximum(out_mag, self.floor * mic_mag)
        
        # Reconstruct signal
        out_fft = out_mag * np.exp(1j * mic_phase)
        out_signal = np.fft.irfft(out_fft, n=self.fft_size)
        
        # Extract frame
        return out_signal[:self.frame_size].astype(np.float32)
    
    def reset(self):
        """Reset internal state."""
        self.ref_magnitude_smooth = None
