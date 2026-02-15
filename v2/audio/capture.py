"""
Audio capture module for microphone and reference (speaker monitor) streams.
Uses sounddevice for cross-platform audio I/O.
Supports PipeWire/PulseAudio monitor sources via pw-record and parec for Linux systems.
"""

import threading
import queue
import numpy as np
import sounddevice as sd
import subprocess
import struct
import shutil
import tempfile
import os
import time
from typing import Optional, Callable, Tuple

# Handle both package and direct execution
try:
    from .. import config
except ImportError:
    import config


# Constants
FLOAT32_BYTES = 4  # Size of float32 in bytes
PWRECORD_HEADER_SIZE = 80  # WAV header size for float32 pw-record output (44-byte standard + 36-byte data chunk header)
PWRECORD_STARTUP_DELAY = 0.4  # Seconds to wait for pw-record to start and write WAV header


def _detect_pipewire_monitor_source() -> Optional[str]:
    """
    Detect available PipeWire/PulseAudio monitor source.
    
    Returns:
        Monitor source name (e.g., 'bluez_output.*.monitor') or None if not found
    """
    try:
        # Check if pactl is available
        if not shutil.which('pactl'):
            return None
        
        # Run pactl list short sources
        result = subprocess.run(
            ['pactl', 'list', 'short', 'sources'],
            capture_output=True,
            text=True,
            timeout=2.0
        )
        
        if result.returncode != 0:
            return None
        
        # Parse output to find monitor sources
        # Format: ID  NAME  DRIVER  SAMPLE_SPEC  STATE
        running_monitor = None
        fallback_monitor = None
        
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            
            # Split on tabs first (pactl uses tab-delimited output)
            parts = line.split('\t')
            
            # Fallback to whitespace split if tab split gives too few parts
            if len(parts) < 3:
                parts = line.split()
            
            if len(parts) < 2:
                continue
            
            source_name = parts[1].strip()
            # State is always the last field
            state = parts[-1].strip()
            
            # Check for .monitor in source name
            if '.monitor' in source_name:
                if state == 'RUNNING':
                    # Prefer a running monitor source
                    running_monitor = source_name
                    break
                elif fallback_monitor is None:
                    # Keep first monitor as fallback
                    fallback_monitor = source_name
        
        return running_monitor or fallback_monitor
    
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        return None


def _get_default_monitor_source() -> Optional[str]:
    """
    Get the monitor source for the default sink.
    This is a simpler and more reliable detection method.
    
    Returns:
        Monitor source name (e.g., 'bluez_output.*.monitor') or None if not found
    """
    try:
        if not shutil.which('pactl'):
            return None
        
        result = subprocess.run(
            ['pactl', 'get-default-sink'],
            capture_output=True,
            text=True,
            timeout=2.0
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip() + '.monitor'
        
        return None
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        return None


class AudioCapture:
    """
    Manages audio capture from microphone and reference (speaker monitor) streams.
    Provides synchronized audio frames for processing.
    Supports both sounddevice and parec-based reference capture.
    """
    
    def __init__(self, mic_device_id: Optional[int] = None, 
                 ref_device_id: Optional[int] = None,
                 ref_capture_method: str = 'auto'):
        """
        Initialize audio capture.
        
        Args:
            mic_device_id: Device ID for microphone (-1 or None for default)
            ref_device_id: Device ID for reference/monitor (-1 or None for default)
            ref_capture_method: Reference capture method ('auto', 'sounddevice', 'parec', 'none')
        """
        self.mic_device_id = mic_device_id if mic_device_id not in (-1, None) else None
        self.ref_device_id = ref_device_id if ref_device_id not in (-1, None) else None
        self.ref_capture_method = ref_capture_method
        
        self.sample_rate = config.DEVICE_RATE
        self.frame_size = config.FRAME_SIZE
        
        # Audio queues
        self.mic_queue = queue.Queue(maxsize=10)
        self.ref_queue = queue.Queue(maxsize=10)
        
        # Streams
        self.mic_stream: Optional[sd.InputStream] = None
        self.ref_stream: Optional[sd.InputStream] = None
        
        # Reference capture subprocess and thread (for parec or pw-record)
        self.ref_process: Optional[subprocess.Popen] = None
        self.ref_thread: Optional[threading.Thread] = None
        self.ref_monitor_source: Optional[str] = None
        self.ref_temp_file: Optional[str] = None
        
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
    
    def _parec_capture_loop(self):
        """Thread loop for capturing audio from parec subprocess."""
        if not self.ref_process:
            return
        
        samples_per_frame = self.frame_size
        bytes_per_frame = samples_per_frame * FLOAT32_BYTES
        
        try:
            # Check self.running with lock to avoid race conditions
            while True:
                with self.lock:
                    if not self.running:
                        break
                
                if not self.ref_process or self.ref_process.poll() is not None:
                    break
                
                # Read one frame worth of audio data
                data = self.ref_process.stdout.read(bytes_per_frame)
                
                if not data or len(data) < bytes_per_frame:
                    break
                
                # Convert bytes to float32 array
                audio = np.frombuffer(data, dtype=np.float32)
                
                # Put in queue (non-blocking)
                try:
                    self.ref_queue.put_nowait(audio)
                except queue.Full:
                    pass  # Drop frame if queue is full
        
        except Exception as e:
            with self.lock:
                if self.on_error and self.running:
                    self.on_error(Exception(f"parec capture error: {e}"))
    
    def _pwrecord_chase_loop(self):
        """Thread loop for chase-reading audio from pw-record WAV file."""
        if not self.ref_temp_file:
            return
        
        samples_per_frame = self.frame_size
        bytes_per_frame = samples_per_frame * FLOAT32_BYTES
        
        try:
            with open(self.ref_temp_file, 'rb') as f:
                # Skip WAV header (pw-record writes standard WAV header for float32 format)
                f.read(PWRECORD_HEADER_SIZE)
                
                # Chase-read loop
                while True:
                    with self.lock:
                        if not self.running:
                            break
                    
                    # Read one frame worth of audio data
                    data = f.read(bytes_per_frame)
                    
                    if len(data) < bytes_per_frame:
                        # Not enough data yet, wait briefly
                        time.sleep(0.005)
                        continue
                    
                    # Convert bytes to float32 array
                    audio = np.frombuffer(data, dtype=np.float32)
                    
                    # Put in queue (non-blocking)
                    try:
                        self.ref_queue.put_nowait(audio)
                    except queue.Full:
                        pass  # Drop frame if queue is full
        
        except Exception as e:
            with self.lock:
                if self.on_error and self.running:
                    self.on_error(Exception(f"pw-record capture error: {e}"))

    
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
                
                # Determine reference capture method
                use_ref_capture = False
                
                if self.ref_capture_method == 'none':
                    # Explicitly disabled
                    pass
                elif self.ref_capture_method == 'sounddevice':
                    # Force sounddevice method
                    if self.ref_device_id is not None:
                        self._start_sounddevice_ref()
                elif self.ref_capture_method == 'parec':
                    # Force parec method
                    use_ref_capture = 'parec'
                else:  # 'auto'
                    # Auto-detect: prefer sounddevice if available, fallback to pw-record/parec
                    if self.ref_device_id is not None:
                        self._start_sounddevice_ref()
                    else:
                        # Try pw-record/parec for PipeWire/PulseAudio monitor
                        use_ref_capture = 'auto'
                
                # Start reference capture if needed
                if use_ref_capture:
                    if use_ref_capture == 'parec':
                        # Force parec only
                        self._start_parec_ref()
                    else:  # 'auto'
                        # Try pw-record first, fall back to parec
                        self._start_pwrecord_ref()
                        # If pw-record didn't start, try parec
                        if not self.ref_process:
                            self._start_parec_ref()

                
                self.running = True
                return True
                
            except Exception as e:
                if self.on_error:
                    self.on_error(e)
                self.stop()
                return False
    
    def _start_sounddevice_ref(self):
        """Start reference capture using sounddevice."""
        self.ref_stream = sd.InputStream(
            device=self.ref_device_id,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            callback=self._ref_callback,
            dtype=np.float32
        )
        self.ref_stream.start()
    
    def _start_parec_ref(self):
        """Start reference capture using parec subprocess."""
        # Check for manual override first, then auto-detect
        monitor_source = config.REF_MONITOR_SOURCE if config.REF_MONITOR_SOURCE else None
        if not monitor_source:
            # Try simple method first, then fall back to parser
            monitor_source = _get_default_monitor_source()
            if not monitor_source:
                monitor_source = _detect_pipewire_monitor_source()
        
        if not monitor_source:
            # No monitor source available
            return
        
        # Check if parec is available
        parec_cmd = shutil.which('parec') or shutil.which('parecord')
        if not parec_cmd:
            return
        
        self.ref_monitor_source = monitor_source
        
        # Start parec subprocess
        # Output: float32le, mono, 48000Hz, raw PCM
        cmd = [
            parec_cmd,
            f'--device={monitor_source}',
            '--format=float32le',
            '--channels=1',
            f'--rate={self.sample_rate}',
            '--raw'
        ]
        
        try:
            # Use larger buffer to reduce context switches (4 frames worth)
            buffer_size = self.frame_size * FLOAT32_BYTES * 4
            
            self.ref_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=buffer_size
            )
            
            # Start capture thread
            self.ref_thread = threading.Thread(
                target=self._parec_capture_loop,
                daemon=True
            )
            self.ref_thread.start()
        
        except Exception as e:
            if self.on_error:
                self.on_error(Exception(f"Failed to start parec: {e}"))
            self.ref_process = None
            self.ref_thread = None
            self.ref_monitor_source = None
    
    def _start_pwrecord_ref(self):
        """Start reference capture using pw-record subprocess with chase-read."""
        # Check if pw-record is available
        if not shutil.which('pw-record'):
            return
        
        # Determine monitor target
        # Priority: manual override > default sink monitor > parser detection
        monitor_target = config.REF_MONITOR_SOURCE if config.REF_MONITOR_SOURCE else None
        if not monitor_target:
            # Try simple method first
            monitor_target = _get_default_monitor_source()
            if not monitor_target:
                # Fall back to parser
                monitor_target = _detect_pipewire_monitor_source()
        
        if not monitor_target:
            # No monitor source available
            return
        
        self.ref_monitor_source = monitor_target
        
        try:
            # Create temporary WAV file
            fd, temp_file = tempfile.mkstemp(suffix='.wav', prefix='pw_chase_')
            os.close(fd)  # Close the file descriptor, we'll open it later
            self.ref_temp_file = temp_file
            
            # Start pw-record subprocess
            # Output: float32, mono, sample_rate Hz, WAV file
            cmd = [
                'pw-record',
                f'--target={monitor_target}',
                '--format=f32',
                '--channels=1',
                f'--rate={self.sample_rate}',
                temp_file
            ]
            
            self.ref_process = subprocess.Popen(
                cmd,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for pw-record to start and write header
            time.sleep(PWRECORD_STARTUP_DELAY)
            
            # Start chase-read thread
            self.ref_thread = threading.Thread(
                target=self._pwrecord_chase_loop,
                daemon=True
            )
            self.ref_thread.start()
        
        except Exception as e:
            if self.on_error:
                self.on_error(Exception(f"Failed to start pw-record: {e}"))
            # Clean up on error
            if self.ref_process:
                try:
                    self.ref_process.terminate()
                    self.ref_process.wait(timeout=1.0)
                except Exception:
                    pass
            if self.ref_temp_file and os.path.exists(self.ref_temp_file):
                try:
                    os.unlink(self.ref_temp_file)
                except Exception:
                    pass
            self.ref_process = None
            self.ref_thread = None
            self.ref_monitor_source = None
            self.ref_temp_file = None

    
    def stop(self):
        """Stop audio capture streams."""
        with self.lock:
            if not self.running:
                return
            
            # Stop microphone stream
            if self.mic_stream:
                try:
                    self.mic_stream.stop()
                    self.mic_stream.close()
                except Exception:
                    pass
                self.mic_stream = None
            
            # Stop reference stream (sounddevice)
            if self.ref_stream:
                try:
                    self.ref_stream.stop()
                    self.ref_stream.close()
                except Exception:
                    pass
                self.ref_stream = None
            
            # Stop reference capture subprocess (parec or pw-record)
            if self.ref_process:
                try:
                    self.ref_process.terminate()
                    # Give it a moment to terminate gracefully
                    try:
                        self.ref_process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        # Force kill if not terminated
                        self.ref_process.kill()
                        self.ref_process.wait()
                except Exception:
                    pass
                self.ref_process = None
            
            # Signal stop to threads AFTER closing streams/processes
            self.running = False
            
            # Wait for reference capture thread to finish
            if self.ref_thread:
                try:
                    self.ref_thread.join(timeout=2.0)
                except Exception:
                    pass
                self.ref_thread = None
            
            # Clean up temp file if it exists
            if self.ref_temp_file:
                try:
                    if os.path.exists(self.ref_temp_file):
                        os.unlink(self.ref_temp_file)
                except Exception:
                    pass
                self.ref_temp_file = None
            
            self.ref_monitor_source = None

            
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
            'ref_capture_method': self.ref_capture_method,
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
        
        # Determine reference device/source name
        if self.ref_monitor_source:
            # Determine method based on temp file existence
            method = 'pw-record' if self.ref_temp_file else 'parec'
            info['ref_device_name'] = f'PipeWire monitor ({self.ref_monitor_source}) via {method}'
        elif self.ref_device_id is not None:
            try:
                device_info = sd.query_devices(self.ref_device_id)
                info['ref_device_name'] = f"sounddevice ({device_info['name']})"
            except Exception:
                info['ref_device_name'] = f'sounddevice (device {self.ref_device_id})'
        else:
            info['ref_device_name'] = 'Not configured'
        
        return info
