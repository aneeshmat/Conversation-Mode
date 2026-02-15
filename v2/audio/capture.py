"""
Audio capture module for microphone and reference (speaker monitor) streams.
Uses sounddevice for cross-platform audio I/O.
"""

import threading
import queue
import numpy as np
import sounddevice as sd
from typing import Optional, Callable, Tuple

# Handle both package and direct execution
try:
    from .. import config
except ImportError:
    import config


class AudioCapture:
    """
    Manages audio capture from microphone and reference (speaker monitor) streams.
    Provides synchronized audio frames for processing.
    """
    
    def __init__(self, mic_device_id: Optional[int] = None, 
                 ref_device_id: Optional[int] = None):
        """
        Initialize audio capture.
        
        Args:
            mic_device_id: Device ID for microphone (-1 or None for default)
            ref_device_id: Device ID for reference/monitor (-1 or None for default)
        """
        self.mic_device_id = mic_device_id if mic_device_id not in (-1, None) else None
        self.ref_device_id = ref_device_id if ref_device_id not in (-1, None) else None
        
        self.sample_rate = config.DEVICE_RATE
        self.frame_size = config.FRAME_SIZE
        
        # Audio queues
        self.mic_queue = queue.Queue(maxsize=10)
        self.ref_queue = queue.Queue(maxsize=10)
        
        # Streams
        self.mic_stream: Optional[sd.InputStream] = None
        self.ref_stream: Optional[sd.InputStream] = None
        
        # State
        self.running = False
        self.lock = threading.Lock()
        
        # Callbacks
        self.on_error: Optional[Callable[[Exception], None]] = None
    
    def _mic_callback(self, indata, frames, time_info, status):
        """Callback for microphone stream."""
        if status:
            if self.on_error:
                self.on_error(Exception(f"Mic stream status: {status}"))
        
        try:
            # Convert to float32 mono
            audio = indata.copy().flatten().astype(np.float32)
            self.mic_queue.put_nowait(audio)
        except queue.Full:
            pass  # Drop frame if queue is full
    
    def _ref_callback(self, indata, frames, time_info, status):
        """Callback for reference stream."""
        if status:
            if self.on_error:
                self.on_error(Exception(f"Ref stream status: {status}"))
        
        try:
            # Convert to float32 mono
            audio = indata.copy().flatten().astype(np.float32)
            self.ref_queue.put_nowait(audio)
        except queue.Full:
            pass  # Drop frame if queue is full
    
    def start(self) -> bool:
        """
        Start audio capture streams.
        
        Returns:
            bool: True if started successfully
        """
        with self.lock:
            if self.running:
                return True
            
            try:
                # Start microphone stream
                self.mic_stream = sd.InputStream(
                    device=self.mic_device_id,
                    channels=1,
                    samplerate=self.sample_rate,
                    blocksize=self.frame_size,
                    callback=self._mic_callback,
                    dtype=np.float32
                )
                self.mic_stream.start()
                
                # Start reference stream (if device specified)
                if self.ref_device_id is not None:
                    self.ref_stream = sd.InputStream(
                        device=self.ref_device_id,
                        channels=1,
                        samplerate=self.sample_rate,
                        blocksize=self.frame_size,
                        callback=self._ref_callback,
                        dtype=np.float32
                    )
                    self.ref_stream.start()
                
                self.running = True
                return True
                
            except Exception as e:
                if self.on_error:
                    self.on_error(e)
                self.stop()
                return False
    
    def stop(self):
        """Stop audio capture streams."""
        with self.lock:
            if not self.running:
                return
            
            if self.mic_stream:
                try:
                    self.mic_stream.stop()
                    self.mic_stream.close()
                except Exception:
                    pass
                self.mic_stream = None
            
            if self.ref_stream:
                try:
                    self.ref_stream.stop()
                    self.ref_stream.close()
                except Exception:
                    pass
                self.ref_stream = None
            
            # Clear queues
            while not self.mic_queue.empty():
                try:
                    self.mic_queue.get_nowait()
                except queue.Empty:
                    break
            
            while not self.ref_queue.empty():
                try:
                    self.ref_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.running = False
    
    def get_frames(self, timeout: float = 0.1) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get next audio frames from mic and reference streams.
        
        Args:
            timeout: Timeout in seconds for waiting on queues
            
        Returns:
            Tuple of (mic_frame, ref_frame), either can be None if not available
        """
        mic_frame = None
        ref_frame = None
        
        try:
            mic_frame = self.mic_queue.get(timeout=timeout)
        except queue.Empty:
            pass
        
        try:
            ref_frame = self.ref_queue.get(timeout=0.001)  # Non-blocking for ref
        except queue.Empty:
            pass
        
        return mic_frame, ref_frame
    
    def is_running(self) -> bool:
        """Check if streams are running."""
        with self.lock:
            return self.running
    
    def get_device_info(self) -> dict:
        """Get information about selected audio devices."""
        info = {
            'mic_device_id': self.mic_device_id,
            'ref_device_id': self.ref_device_id,
            'sample_rate': self.sample_rate,
            'frame_size': self.frame_size
        }
        
        try:
            if self.mic_device_id is not None:
                info['mic_device_name'] = sd.query_devices(self.mic_device_id)['name']
            else:
                info['mic_device_name'] = sd.query_devices(kind='input')['name']
        except Exception:
            info['mic_device_name'] = 'Unknown'
        
        try:
            if self.ref_device_id is not None:
                info['ref_device_name'] = sd.query_devices(self.ref_device_id)['name']
            else:
                info['ref_device_name'] = 'Not configured'
        except Exception:
            info['ref_device_name'] = 'Unknown'
        
        return info
