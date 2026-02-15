"""
Python ctypes wrapper for SpeexDSP-based AEC with fallback.
"""

import ctypes
import numpy as np
import os
import platform
from typing import Optional
from enum import Enum
from .. import config


class AECStatus(Enum):
    """AEC implementation status."""
    SPEEX_AVAILABLE = "SpeexDSP"
    FALLBACK_ACTIVE = "Fallback"
    DISABLED = "Disabled"


class AECSpeex:
    """SpeexDSP-based AEC wrapper using ctypes."""
    
    def __init__(self, frame_size: int, filter_length: int, sample_rate: int):
        """
        Initialize SpeexDSP AEC.
        
        Args:
            frame_size: Samples per frame
            filter_length: AEC filter length in samples
            sample_rate: Audio sample rate in Hz
        """
        self.frame_size = frame_size
        self.filter_length = filter_length
        self.sample_rate = sample_rate
        self.state = None
        
        # Find and load the shared library
        lib_path = self._find_library()
        if not lib_path:
            raise RuntimeError("SpeexDSP library not found")
        
        try:
            self.lib = ctypes.CDLL(lib_path)
            self._setup_functions()
            
            # Create AEC state
            self.state = self.lib.aec_create(
                ctypes.c_int(frame_size),
                ctypes.c_int(filter_length),
                ctypes.c_int(sample_rate)
            )
            
            if not self.state:
                raise RuntimeError("Failed to create AEC state")
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SpeexDSP: {e}")
    
    def _find_library(self) -> Optional[str]:
        """Find the SpeexDSP library."""
        # Determine library extension based on platform
        system = platform.system()
        if system == "Linux":
            lib_name = "libaec_speex.so"
        elif system == "Darwin":
            lib_name = "libaec_speex.dylib"
        elif system == "Windows":
            lib_name = "libaec_speex.dll"
        else:
            return None
        
        # Check in aec directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(base_dir, lib_name)
        
        if os.path.exists(lib_path):
            return lib_path
        
        return None
    
    def _setup_functions(self):
        """Setup ctypes function signatures."""
        # aec_create
        self.lib.aec_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.aec_create.restype = ctypes.c_void_p
        
        # aec_process
        self.lib.aec_process.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.int16, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int16, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int16, flags='C_CONTIGUOUS'),
            ctypes.c_int
        ]
        self.lib.aec_process.restype = None
        
        # aec_reset
        self.lib.aec_reset.argtypes = [ctypes.c_void_p]
        self.lib.aec_reset.restype = None
        
        # aec_destroy
        self.lib.aec_destroy.argtypes = [ctypes.c_void_p]
        self.lib.aec_destroy.restype = None
    
    def process(self, ref_frame: np.ndarray, mic_frame: np.ndarray) -> np.ndarray:
        """
        Process audio frame through AEC.
        
        Args:
            ref_frame: Reference signal (float32, range -1 to 1)
            mic_frame: Microphone signal (float32, range -1 to 1)
            
        Returns:
            Echo-cancelled signal (float32)
        """
        if self.state is None:
            return mic_frame.copy()
        
        # Convert float32 to int16
        ref_int16 = (ref_frame * 32767.0).astype(np.int16)
        mic_int16 = (mic_frame * 32767.0).astype(np.int16)
        out_int16 = np.zeros(self.frame_size, dtype=np.int16)
        
        # Process
        self.lib.aec_process(
            self.state,
            ref_int16,
            mic_int16,
            out_int16,
            ctypes.c_int(self.frame_size)
        )
        
        # Convert back to float32
        return out_int16.astype(np.float32) / 32767.0
    
    def reset(self):
        """Reset AEC state."""
        if self.state:
            self.lib.aec_reset(self.state)
    
    def close(self):
        """Cleanup resources."""
        if self.state:
            self.lib.aec_destroy(self.state)
            self.state = None


class AECProcessor:
    """
    Main AEC processor that handles SpeexDSP with fallback.
    """
    
    def __init__(self, frame_size: int, sample_rate: int, 
                 filter_length: Optional[int] = None, enabled: bool = True):
        """
        Initialize AEC processor.
        
        Args:
            frame_size: Samples per frame
            sample_rate: Audio sample rate in Hz
            filter_length: AEC filter length (None = use config default)
            enabled: Whether to enable AEC
        """
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.filter_length = filter_length or config.AEC_FILTER_LENGTH
        self.enabled = enabled
        
        self.aec_impl = None
        self.status = AECStatus.DISABLED
        
        if enabled:
            self._initialize()
    
    def _initialize(self):
        """Initialize AEC implementation (try SpeexDSP, fallback if needed)."""
        try:
            # Try SpeexDSP
            self.aec_impl = AECSpeex(self.frame_size, self.filter_length, self.sample_rate)
            self.status = AECStatus.SPEEX_AVAILABLE
            if config.DEBUG:
                print("[AEC] Using SpeexDSP")
        except Exception as e:
            # Fallback to Python implementation
            if config.DEBUG:
                print(f"[AEC] SpeexDSP unavailable ({e}), using fallback")
            
            try:
                from .aec_fallback import AECFallback
                self.aec_impl = AECFallback(self.frame_size, self.sample_rate)
                self.status = AECStatus.FALLBACK_ACTIVE
            except Exception as e2:
                if config.DEBUG:
                    print(f"[AEC] Fallback failed ({e2}), AEC disabled")
                self.enabled = False
                self.status = AECStatus.DISABLED
    
    def process(self, ref_frame: Optional[np.ndarray], 
                mic_frame: np.ndarray) -> np.ndarray:
        """
        Process audio through AEC.
        
        Args:
            ref_frame: Reference signal (speaker output), None if not available
            mic_frame: Microphone signal
            
        Returns:
            Processed signal (echo-cancelled if AEC enabled and ref available)
        """
        if not self.enabled or self.aec_impl is None or ref_frame is None:
            return mic_frame.copy()
        
        return self.aec_impl.process(ref_frame, mic_frame)
    
    def set_enabled(self, enabled: bool):
        """Enable or disable AEC."""
        if enabled and not self.enabled:
            self.enabled = True
            if self.aec_impl is None:
                self._initialize()
        elif not enabled and self.enabled:
            self.enabled = False
    
    def get_status(self) -> AECStatus:
        """Get current AEC status."""
        return self.status
    
    def reset(self):
        """Reset AEC state."""
        if self.aec_impl and hasattr(self.aec_impl, 'reset'):
            self.aec_impl.reset()
    
    def close(self):
        """Cleanup resources."""
        if self.aec_impl and hasattr(self.aec_impl, 'close'):
            self.aec_impl.close()
        self.aec_impl = None
