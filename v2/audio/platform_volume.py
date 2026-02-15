"""
Platform-abstracted volume control.
Defines an abstract base class and Linux implementation.
"""

import subprocess
import shutil
import re
from abc import ABC, abstractmethod
from typing import Optional


class VolumeController(ABC):
    """Abstract base class for platform-specific volume control."""
    
    @abstractmethod
    def get_volume(self) -> int:
        """
        Get current system volume as percentage.
        
        Returns:
            int: Volume percentage (0-100), or -1 if error
        """
        pass
    
    @abstractmethod
    def set_volume(self, percent: int) -> bool:
        """
        Set system volume to specified percentage.
        
        Args:
            percent: Target volume percentage (0-100)
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass


class LinuxVolumeController(VolumeController):
    """Linux implementation using pactl (PulseAudio/PipeWire) with amixer fallback."""
    
    def __init__(self):
        self._use_pactl = shutil.which("pactl") is not None
        self._use_amixer = shutil.which("amixer") is not None
        
        if not self._use_pactl and not self._use_amixer:
            raise RuntimeError("Neither pactl nor amixer found. Cannot control volume.")
    
    def _parse_pactl_volume(self, stdout: str) -> int:
        """Parse pactl volume output to extract percentage."""
        vals = [int(m.group(1)) for m in re.finditer(r'(\d+)%', stdout)]
        if vals:
            return int(round(sum(vals) / len(vals)))
        return -1
    
    def get_volume(self) -> int:
        """Get current volume using pactl or amixer."""
        # Try pactl first
        if self._use_pactl:
            try:
                result = subprocess.run(
                    ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                    capture_output=True,
                    text=True,
                    timeout=1.5
                )
                if result.returncode == 0:
                    vol = self._parse_pactl_volume(result.stdout)
                    if vol >= 0:
                        return vol
            except Exception:
                pass
        
        # Fallback to amixer
        if self._use_amixer:
            try:
                result = subprocess.run(
                    ["amixer", "get", "Master"],
                    capture_output=True,
                    text=True,
                    timeout=1.5
                )
                if result.returncode == 0:
                    match = re.search(r'\[(\d+)%\]', result.stdout)
                    if match:
                        return int(match.group(1))
            except Exception:
                pass
        
        return -1
    
    def set_volume(self, percent: int) -> bool:
        """Set volume using pactl or amixer."""
        percent = max(0, min(150, int(percent)))  # Clamp to 0-150%
        
        # Try pactl first
        if self._use_pactl:
            try:
                result = subprocess.run(
                    ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"],
                    capture_output=True,
                    text=True,
                    timeout=1.5
                )
                return result.returncode == 0
            except Exception:
                pass
        
        # Fallback to amixer
        if self._use_amixer:
            try:
                percent = max(0, min(100, int(percent)))  # amixer limited to 0-100%
                result = subprocess.run(
                    ["amixer", "sset", "Master", f"{percent}%"],
                    capture_output=True,
                    text=True,
                    timeout=1.5
                )
                return result.returncode == 0
            except Exception:
                pass
        
        return False


def get_volume_controller() -> Optional[VolumeController]:
    """
    Factory function to get appropriate volume controller for the platform.
    
    Returns:
        VolumeController instance for the current platform, or None if unsupported
    """
    import platform
    
    system = platform.system()
    
    if system == "Linux":
        try:
            return LinuxVolumeController()
        except RuntimeError:
            return None
    
    # TODO: Add Windows implementation using pycaw
    # TODO: Add macOS implementation using osascript
    # TODO: Add iOS implementation
    # TODO: Add Android implementation
    
    return None
