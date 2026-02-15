"""
Base class for audio analysis plugins.
Future extensions (e.g., YamNET classification) can inherit from this.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class AudioAnalysisPlugin(ABC):
    """
    Abstract base class for audio analysis plugins.
    
    Plugins can perform additional analysis on the audio stream, such as:
    - Music detection
    - Speech/singing classification
    - Environmental sound classification
    - Emotion detection
    - Speaker identification
    """
    
    def __init__(self, sample_rate: int, frame_size: int):
        """
        Initialize plugin.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_size: Frame size in samples
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
    
    @abstractmethod
    def process(self, audio_frame: np.ndarray) -> Dict[str, Any]:
        """
        Process an audio frame and return analysis results.
        
        Args:
            audio_frame: Audio samples (float32, normalized to -1 to 1)
            
        Returns:
            Dictionary of analysis results (plugin-specific)
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset plugin state."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get plugin description."""
        pass


# Example plugin implementation (for reference)
class ExampleMusicDetectorPlugin(AudioAnalysisPlugin):
    """
    Example plugin that detects music vs speech.
    This is a placeholder - real implementation would use YamNET or similar.
    """
    
    def __init__(self, sample_rate: int, frame_size: int):
        super().__init__(sample_rate, frame_size)
        self.music_probability = 0.0
    
    def process(self, audio_frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze audio frame for music content.
        
        Returns:
            Dict with 'music_probability' and 'classification' keys
        """
        # Placeholder: use simple spectral features as proxy
        # Real implementation would use a trained classifier
        
        # Compute FFT
        spectrum = np.abs(np.fft.rfft(audio_frame))
        
        # Simple heuristic: music tends to have more energy in mid frequencies
        low_energy = np.sum(spectrum[:len(spectrum)//4])
        mid_energy = np.sum(spectrum[len(spectrum)//4:3*len(spectrum)//4])
        high_energy = np.sum(spectrum[3*len(spectrum)//4:])
        
        total_energy = low_energy + mid_energy + high_energy
        
        if total_energy > 1e-6:
            mid_ratio = mid_energy / total_energy
            # Music typically has more balanced spectrum
            self.music_probability = min(1.0, mid_ratio * 2.0)
        else:
            self.music_probability = 0.0
        
        classification = "music" if self.music_probability > 0.5 else "speech"
        
        return {
            'music_probability': self.music_probability,
            'classification': classification,
            'low_energy': low_energy,
            'mid_energy': mid_energy,
            'high_energy': high_energy
        }
    
    def reset(self):
        """Reset state."""
        self.music_probability = 0.0
    
    def get_name(self) -> str:
        return "Music Detector"
    
    def get_description(self) -> str:
        return "Detects music vs speech in audio stream"
