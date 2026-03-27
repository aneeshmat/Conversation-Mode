"""
Auto Ducking with Silero VAD - Version 3

This module implements real-time audio ducking based on voice activity detection
using the Silero VAD model. It automatically lowers system audio volume when
speech is detected and restores it after a configurable delay.

Features:
- Cross-platform support (Windows and Linux)
- Real-time voice activity detection using Silero VAD
- Thread-safe volume control with smart timer management
- Low-latency audio processing (<50ms)
- Configurable ducking parameters
"""

import time
import threading
import torch
import numpy as np
import sounddevice as sd
import platform
import sys

# Platform-specific imports
if platform.system() == 'Windows':
    try:
        from ctypes import POINTER, cast
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        WINDOWS_AVAILABLE = True
    except ImportError:
        WINDOWS_AVAILABLE = False
        print("Warning: pycaw not available. Volume control disabled on Windows.")
elif platform.system() == 'Linux':
    import subprocess
    LINUX_AVAILABLE = True
else:
    print(f"Warning: Unsupported platform {platform.system()}")
    sys.exit(1)


class AudioDucker:
    """Manages audio ducking based on voice activity detection."""
    
    def __init__(self, 
                 sample_rate=16000, 
                 frame_size=512,
                 duck_duration=1.5,
                 duck_percentage=50,
                 vad_threshold=0.5):
        """
        Initialize the AudioDucker.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
            frame_size: Number of samples per frame (default: 512)
            duck_duration: Duration in seconds to keep volume ducked (default: 1.5)
            duck_percentage: Percentage of original volume when ducked (default: 50)
            vad_threshold: VAD confidence threshold (0.0-1.0, default: 0.5)
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.duck_duration = duck_duration
        self.duck_percentage = duck_percentage
        self.vad_threshold = vad_threshold
        
        self.prev_state = 'SILENCE'
        self.duck_lock = threading.Lock()
        self.duck_timer = None
        
        # Load Silero VAD model
        print("Loading Silero VAD model...")
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.get_speech_timestamps = utils[0]
        print("Silero VAD model loaded successfully.")
        
        # Initialize platform-specific volume control
        self._initialize_volume_control()
    
    def _initialize_volume_control(self):
        """Initialize platform-specific volume control."""
        self.platform = platform.system()
        
        if self.platform == 'Windows' and WINDOWS_AVAILABLE:
            self._init_windows_volume()
        elif self.platform == 'Linux':
            self._init_linux_volume()
        else:
            raise RuntimeError(f"Volume control not supported on {self.platform}")
    
    def _init_windows_volume(self):
        """Initialize Windows volume control using pycaw."""
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
        
        # Save original volume level (0.0 to 1.0)
        self.original_volume = self.volume_interface.GetMasterVolumeLevelScalar()
        self.ducked_volume = self.original_volume * (self.duck_percentage / 100.0)
        
        print(f"Windows volume control initialized.")
        print(f"Original volume: {self.original_volume:.2f}")
        print(f"Ducked volume: {self.ducked_volume:.2f}")
    
    def _init_linux_volume(self):
        """Initialize Linux volume control using pactl."""
        self.original_volume = self._get_linux_volume()
        self.ducked_volume = int(self.original_volume * (self.duck_percentage / 100.0))
        
        print(f"Linux volume control initialized.")
        print(f"Original volume: {self.original_volume}%")
        print(f"Ducked volume: {self.ducked_volume}%")
    
    def _get_linux_volume(self):
        """Get current volume percentage (0-100) using pactl."""
        try:
            result = subprocess.run(
                ['pactl', 'get-sink-volume', '@DEFAULT_SINK@'],
                capture_output=True,
                text=True,
                check=True
            )
            # Parse output like: "Volume: front-left: 65536 / 100% / 0.00 dB"
            for part in result.stdout.split():
                if '%' in part:
                    return int(part.rstrip('%'))
        except subprocess.CalledProcessError as e:
            print(f"Error getting volume: {e}")
        except FileNotFoundError:
            print("Error: pactl not found. Please install PulseAudio or PipeWire.")
        return 50  # Default fallback
    
    def _set_volume_windows(self, volume_level):
        """Set Windows volume level (0.0 to 1.0)."""
        self.volume_interface.SetMasterVolumeLevelScalar(volume_level, None)
    
    def _set_volume_linux(self, volume_percent):
        """Set Linux volume percentage (0-100)."""
        try:
            subprocess.run(
                ['pactl', 'set-sink-volume', '@DEFAULT_SINK@', f'{volume_percent}%'],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error setting volume: {e}")
        except FileNotFoundError:
            print("Error: pactl not found. Please install PulseAudio or PipeWire.")
    
    def restore_volume(self):
        """Restore the original system volume."""
        with self.duck_lock:
            if self.platform == 'Windows':
                self._set_volume_windows(self.original_volume)
            elif self.platform == 'Linux':
                self._set_volume_linux(self.original_volume)
            print("🔊 Volume restored")
            self.duck_timer = None
    
    def duck_volume(self):
        """Duck (lower) the system volume."""
        with self.duck_lock:
            # Cancel any existing timer
            if self.duck_timer and self.duck_timer.is_alive():
                self.duck_timer.cancel()
            
            # Lower volume
            if self.platform == 'Windows':
                self._set_volume_windows(self.ducked_volume)
            elif self.platform == 'Linux':
                self._set_volume_linux(self.ducked_volume)
            
            print(f"🔉 Volume ducked for {self.duck_duration}s")
            
            # Start timer to restore volume after duck_duration
            self.duck_timer = threading.Timer(self.duck_duration, self.restore_volume)
            self.duck_timer.start()
    
    def audio_callback(self, indata, frames, time_info, status):
        """
        Audio callback function for processing incoming audio frames.
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Timing information
            status: Stream status
        """
        if status:
            print(f"Stream status: {status}")
        
        if frames != self.frame_size:
            print(f"Warning: Expected {self.frame_size} frames but got {frames}")
            return
        
        # Convert audio frame to tensor
        audio_frame = indata[:, 0].copy()
        audio_tensor = torch.from_numpy(audio_frame)
        
        # Run VAD inference
        speech_probs = self.model(audio_tensor, self.sample_rate).flatten()
        is_speech = torch.any(speech_probs > self.vad_threshold).item()
        
        # State management and volume ducking
        if is_speech and self.prev_state != 'SPEAKING':
            timestamp = time_info.inputBufferAdcTime if time_info else 0
            print(f"{timestamp:.2f}s — 🎤 SPEAKING detected")
            self.prev_state = 'SPEAKING'
            self.duck_volume()
        elif not is_speech and self.prev_state != 'SILENCE':
            timestamp = time_info.inputBufferAdcTime if time_info else 0
            print(f"{timestamp:.2f}s — 🤐 SILENCE")
            self.prev_state = 'SILENCE'
    
    def run(self):
        """Start the audio ducking system."""
        print("=" * 60)
        print("Auto Ducking with Silero VAD - Version 3")
        print("=" * 60)
        print(f"Platform: {self.platform}")
        print(f"Sample Rate: {self.sample_rate} Hz")
        print(f"Frame Size: {self.frame_size} samples")
        print(f"Duck Duration: {self.duck_duration}s")
        print(f"Duck Percentage: {self.duck_percentage}%")
        print(f"VAD Threshold: {self.vad_threshold}")
        print("=" * 60)
        print("\n🎙️  Real-time VAD with volume ducking started.")
        print("Speak into the microphone to trigger audio ducking.")
        print("Press Ctrl+C to stop.\n")
        
        try:
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.frame_size
            ):
                while True:
                    sd.sleep(100)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping...")
            self.restore_volume()
            print("✅ Auto ducking stopped successfully.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            self.restore_volume()
            raise


def main():
    """Main entry point for the auto ducking system."""
    # Create and run the audio ducker with default settings
    ducker = AudioDucker(
        sample_rate=16000,
        frame_size=512,
        duck_duration=1.5,
        duck_percentage=50,
        vad_threshold=0.5
    )
    ducker.run()


if __name__ == "__main__":
    main()
