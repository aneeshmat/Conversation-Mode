# Conversation Mode (Linux) - WebRTC AEC + VAD + Ducking

## Overview
This project implements a robust real-time audio pipeline for Linux, featuring:
- Acoustic Echo Cancellation (AEC) using WebRTC
- Voice Activity Detection (VAD) using Silero (Torch/ONNX)
- Smooth system volume ducking during speech
- Device auto-selection (prefers 'pulse')
- Tkinter GUI for control and status

## Key Features
- **AEC:** Uses WebRTC AEC via Python wrapper for robust echo cancellation.
- **VAD:** Uses Silero VAD (Torch or ONNX fallback) for accurate speech detection.
- **Ducking:** System volume is smoothly reduced during detected speech, restored after silence.
- **Device Handling:** Auto-selects 'pulse' or first available input device for mic/ref.
- **GUI:** Tkinter-based interface for start/stop and status display.

## Debugging & Fixes
### 1. AEC Integration
- Initial pipeline used SpeexDSP AEC (C shared lib + ctypes).
- Migrated to WebRTC AEC for improved robustness.
- Fixed Python 3.12 import issues and .so file naming.
- Updated API usage to match WebRTC Python package.
- Patched device selection to prefer 'pulse' and fall back to first available input.

### 2. VAD Integration
- Integrated Silero VAD (Torch) for speech detection.
- Added ONNX fallback for environments without Torch.
- Implemented smoothing and hysteresis for robust VAD gating.

### 3. Volume Ducking
- Restored system volume ducking using pactl (PipeWire/PulseAudio) and amixer (ALSA) fallback.
- Patched DuckController to perform actual volume changes during speech and restore after silence.
- Ensured user volume changes are respected while ducked.

### 4. Device Robustness
- Patched device selection logic to auto-select 'pulse' or first available input device.
- Added error handling for missing devices.

### 5. Debugging & Error Fixes
- Cleaned up undefined/unused variables and functions.
- Used get_errors to verify code was error-free after each patch.
- Fixed issues with device selection, AEC loading, and volume control.

### 6. Gating Logic Improvements
- Implemented robust gating logic to reduce false ducking from speaker output.
- Tuned RMS thresholds and VAD attack/release for environment.
- Experimented with aggressive double-talk suppression and GUI controls (later reverted for simplicity).

### 7. ML-Based Separation (Experimental)
- Explored Demucs for offline stem separation (not suitable for real-time).
- Determined that real-time VAD/AEC is best achieved with lightweight models (Silero, WebRTC).

## Dependencies
- numpy
- sounddevice
- packaging
- torch, torchvision, torchaudio (for VAD)
- onnxruntime (for VAD fallback)
- webrtc-audio-processing (for AEC)
- tkinter (for GUI)

## Usage
1. Install dependencies:
   ```bash
   pip install numpy sounddevice packaging torch torchvision torchaudio onnxruntime webrtc-audio-processing
   ```
2. Run the script:
   ```bash
   python3 conversation_mode_aec_linux_v8_webrtc.py
   ```
3. Use the GUI to start/stop conversation mode and monitor speech probability.

## Lessons Learned
- WebRTC AEC is robust but not as advanced as Zoom; ML-based post-filters can further improve echo suppression.
- Silero VAD is fast and accurate for real-time speech detection.
- System volume ducking is best controlled via pactl/amixer for Linux.
- Device auto-selection and error handling are critical for user-friendly operation.
- Real-time ML-based stem separation is not practical on CPU; stick with lightweight VAD/AEC for live pipelines.

## Future Improvements

This documentation summarizes the development, debugging, and feature integration for conversation_mode_aec_linux_v8_webrtc.py. For further improvements or integration, see the project README or contact the developer.

## Custom AEC Integration: Setup & Modifications

### 1. Clone and Build EC Repository
1. Clone the EC repository:
   ```bash
   git clone https://github.com/voice-engine/ec.git
   cd ec
   ```
2. Build the EC C library:
   ```bash
   make
   cd src
   gcc -fPIC -shared -o libec.so ec.c audio.c fifo.c pa_ringbuffer.c util.c -lspeexdsp -lm -lpthread -lrt -lasound
   ```
   This produces `libec.so` in the `src` directory.

### 2. Create/Modify ec.h (C Header)
Add or update `ec.h` in the EC repo to ensure exported functions for ctypes:
```c
#ifdef __cplusplus
extern "C" {
#endif

#define EXPORT __attribute__((visibility("default")))

EXPORT void *ec_create(int rate, int frame_size);
EXPORT void ec_destroy(void *obj);
EXPORT void ec_process(void *obj, short *mic, short *ref, short *out);

#ifdef __cplusplus
}
#endif
```

### 3. Create ec_wrapper.py (Python ctypes Wrapper)
Create `ec_wrapper.py` to interface with the C library:
```python
import numpy as np
import ctypes
import os

class EC:
   def __init__(self, sample_rate, frame_size):
      lib_path = os.path.join(os.path.dirname(__file__), '../ec/src/libec.so')
      self.lib = ctypes.CDLL(lib_path)
      self.frame_size = frame_size
      self.sample_rate = sample_rate
      self.obj = self.lib.ec_create(sample_rate, frame_size)
      self.lib.ec_create.restype = ctypes.c_void_p
      self.lib.ec_destroy.argtypes = [ctypes.c_void_p]
      self.lib.ec_process.argtypes = [ctypes.c_void_p,
                              ctypes.POINTER(ctypes.c_short),
                              ctypes.POINTER(ctypes.c_short),
                              ctypes.POINTER(ctypes.c_short)]

   def echo_cancel(self, mic, ref):
      out = np.zeros(self.frame_size, dtype=np.int16)
      mic_ptr = mic.ctypes.data_as(ctypes.POINTER(ctypes.c_short))
      ref_ptr = ref.ctypes.data_as(ctypes.POINTER(ctypes.c_short))
      out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_short))
      self.lib.ec_process(self.obj, mic_ptr, ref_ptr, out_ptr)
      return out

   def __del__(self):
      if hasattr(self, 'obj') and self.obj:
         self.lib.ec_destroy(self.obj)
```

### 4. Integration Steps
- Ensure `libec.so` is built and accessible from your Python code.
- Place `ec_wrapper.py` in your project (adjust path to `libec.so` as needed).
- Use the `EC` class to perform echo cancellation on int16 numpy arrays.
- Update your main pipeline to call `EC.echo_cancel(mic, ref)` for each frame.

### 5. Troubleshooting
- If `libec.so` fails to load, check build steps and library path.
- Ensure all exported functions in `ec.h` match those used in `ec_wrapper.py`.
- Use `ctypes` error handling for debugging library loading issues.

---
This section details the full setup and integration process for custom AEC using the EC repository, including all necessary file modifications and wrapper creation.
