# Conversation Mode v2

**Real-time voice-activity-driven audio ducking with advanced echo cancellation and smart volume tracking.**

## Overview

Conversation Mode v2 is a complete rewrite of the original Conversation Mode software with a clean, modular architecture. It automatically lowers system volume when speech is detected and restores it when speech ends, making it easy to have conversations while music or other audio is playing.

### Use Case

> "I am sitting at my desk with my girlfriend and we are blasting music from a speaker. When either of us is talking, the system volume decreases such that we can hear each other. When we are done talking, the system volume is restored."

## Key Features

### 🎯 Smart Auto-Ducking with Dynamic Baseline Tracking

- **Continuous speech tracking**: Unlike the old version with a fixed timer, v2 tracks speech state continuously and only restores volume when speech actually stops
- **Dynamic baseline tracking**: Detects when you change the volume while ducking is active and updates the baseline accordingly
- **Smooth transitions**: Volume changes are smoothly interpolated to avoid jarring jumps
- **Minimum volume floor**: Prevents complete silence by enforcing a minimum ducked volume

### 🔇 Acoustic Echo Cancellation (AEC)

- **SpeexDSP integration**: Uses industry-standard SpeexDSP library for high-quality echo cancellation
- **Fallback support**: Pure Python spectral subtraction fallback when SpeexDSP is unavailable
- **Real-time processing**: Removes speaker output from microphone input so music doesn't trigger ducking

### 🎤 Voice Activity Detection (VAD)

- **Silero VAD**: State-of-the-art VAD model with sub-50ms latency
- **EMA smoothing**: Exponential moving average smoothing for stable detection
- **Hysteresis**: Separate on/off thresholds prevent rapid toggling
- **Hold timer**: Configurable hold period after speech ends before restoring volume

### 🖥️ Cross-Platform Architecture

- **Platform abstraction**: Clean abstract base class for volume control
- **Linux support**: Uses pactl (PulseAudio/PipeWire) with amixer fallback
- **Future-ready**: Designed for easy addition of Windows, macOS, iOS, Android support

### 📊 Real-Time GUI

- Start/Stop control
- Live VAD probability display with progress bar
- Speech state indicator
- Ducking state indicator  
- Current and baseline volume display
- AEC status indicator
- Audio device information

### 🔧 Modular Architecture

Clean separation of concerns with well-defined interfaces:

```
v2/
├── audio/              # Audio capture and volume control
├── aec/                # Echo cancellation
├── vad/                # Voice activity detection
├── ducking/            # Smart volume ducking
├── pipeline/           # Main processing pipeline
├── gui/                # User interface
└── plugins/            # Extensible plugin system
```

## Requirements

### System Requirements

- **OS**: Linux (Ubuntu, Debian, etc.) - Windows/macOS coming soon
- **Python**: 3.8 or higher
- **Audio**: PulseAudio or PipeWire

### Python Dependencies

```bash
pip install -r requirements.txt
```

- `torch>=2.0` - For Silero VAD
- `numpy>=1.24` - Numerical operations
- `sounddevice>=0.4` - Cross-platform audio I/O

Optional:
- `onnxruntime>=1.16` - For ONNX VAD fallback (if PyTorch unavailable)

### System Dependencies (for SpeexDSP AEC)

**Debian/Ubuntu:**
```bash
sudo apt install libspeexdsp-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install speexdsp-devel
```

**Arch Linux:**
```bash
sudo pacman -S speexdsp
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aneeshmat/Conversation-Mode.git
   cd Conversation-Mode/v2
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Build SpeexDSP AEC library (optional but recommended):**
   ```bash
   sudo apt install libspeexdsp-dev  # On Debian/Ubuntu
   make
   ```
   
   If SpeexDSP is not available, the system will fall back to a Python-based spectral subtraction AEC.

4. **Configure audio devices (optional):**
   
   List available audio devices:
   ```bash
   python -c "import sounddevice as sd; print(sd.query_devices())"
   ```
   
   Set environment variables:
   ```bash
   export MIC_DEVICE_ID=5      # Your microphone device ID
   export REF_DEVICE_ID=12     # Your monitor/loopback device ID
   ```
   
   **Linux PipeWire/PulseAudio Users:**
   
   For systems using PipeWire (e.g., modern Ubuntu, Fedora), the system can automatically detect and use monitor sources via `parec`:
   
   ```bash
   # Auto-detect monitor source (default behavior)
   python main.py
   
   # Force parec method
   REF_CAPTURE_METHOD=parec python main.py
   
   # View available monitor sources
   pactl list short sources | grep monitor
   ```
   
   This solves issues where sounddevice cannot see PipeWire monitor sources (e.g., Bluetooth speaker monitors). Install PulseAudio utilities if needed:
   ```bash
   sudo apt install pulseaudio-utils
   ```

## Usage

### Basic Usage

```bash
cd v2
python main.py
```

### With Custom Device IDs

```bash
MIC_DEVICE_ID=5 REF_DEVICE_ID=12 python main.py
```

### With PipeWire Monitor (Recommended for Linux)

```bash
# Auto-detect and use PipeWire monitor source
python main.py

# Or force parec method
REF_CAPTURE_METHOD=parec python main.py
```

### With Debug Logging

```bash
DEBUG=1 python main.py
```

### GUI Controls

- **Start/Stop Button**: Enable/disable the entire pipeline
- **AEC Toggle**: Enable/disable acoustic echo cancellation
- **Status Display**: Real-time monitoring of:
  - VAD probability (0.0 - 1.0)
  - Speech state (Active/Inactive)
  - Ducking state (Active/Inactive)
  - Current system volume
  - Baseline volume (when ducked)
  - AEC status (SpeexDSP/Fallback/Disabled)

## Configuration

All tunable parameters are in `config.py`:

### Audio Configuration
- `SAMPLE_RATE = 16000` - VAD processing rate
- `DEVICE_RATE = 48000` - Hardware I/O rate
- `FRAME_SIZE = 1024` - Samples per frame
- `MIC_DEVICE_ID` - Microphone device ID (-1 for default)
- `REF_DEVICE_ID` - Reference/monitor device ID (-1 for default)
- `REF_CAPTURE_METHOD` - Reference capture method:
  - `'auto'` (default) - Use sounddevice if REF_DEVICE_ID is set, otherwise try parec
  - `'sounddevice'` - Force sounddevice (requires valid REF_DEVICE_ID)
  - `'parec'` - Force parec for PipeWire/PulseAudio monitor sources
  - `'none'` - Disable reference capture (no AEC)

### Ducking Configuration
- `DUCK_RATIO = 0.5` - Duck to 50% of baseline
- `DUCK_MIN_PERCENT = 5` - Minimum ducked volume
- `DUCK_HOLD_SEC = 1.5` - Hold after speech stops
- `DUCK_SMOOTH_STEPS = 8` - Smooth transition steps
- `DUCK_SMOOTH_STEP_MS = 25` - Time between steps

### VAD Configuration
- `VAD_ON_THRESHOLD = 0.65` - Threshold to trigger speech
- `VAD_OFF_THRESHOLD = 0.35` - Threshold to end speech
- `VAD_EMA_ALPHA = 0.3` - EMA smoothing factor
- `VAD_HOLD_FRAMES = 10` - Hold frames after drop

### AEC Configuration
- `AEC_FILTER_LENGTH = 1024` - Filter length in samples
- `AEC_DELAY = 0` - Delay compensation (0 = auto)

## Architecture

### Audio Processing Pipeline

```
1. Microphone Input (48kHz)
          ↓
2. Reference Input (48kHz, speaker monitor)
          ↓
3. Acoustic Echo Cancellation
          ↓
4. DC-Blocking High-Pass Filter
          ↓
5. Gain Boost
          ↓
6. Resample to 16kHz
          ↓
7. Silero VAD
          ↓
8. EMA Smoothing + Hysteresis
          ↓
9. Duck Controller
          ↓
10. Volume Control
```

### Smart Ducking Logic

```python
on each frame:
  if speech_detected:
    last_speech_time = now
    if not ducked:
      # Start ducking
      baseline = get_current_volume()
      target = max(baseline * DUCK_RATIO, DUCK_MIN_PERCENT)
      smooth_set_volume(baseline -> target)
      ducked = True
    else:
      # Check if user changed volume
      current = get_current_volume()
      if abs(current - expected_ducked_volume) > threshold:
        # Update baseline to user's new preference
        new_baseline = current
        new_target = max(new_baseline * DUCK_RATIO, DUCK_MIN_PERCENT)
        smooth_set_volume(current -> new_target)
        baseline = new_baseline
  else:
    if ducked and (now - last_speech_time) >= DUCK_HOLD_SEC:
      # Restore to baseline
      smooth_set_volume(current -> baseline)
      ducked = False
```

## Troubleshooting

### Audio Device Configuration

If you see "Failed to start audio capture", you may need to specify device IDs:

1. List devices:
   ```bash
   python -c "import sounddevice as sd; print(sd.query_devices())"
   ```

2. Set environment variables:
   ```bash
   export MIC_DEVICE_ID=<your_mic_id>
   export REF_DEVICE_ID=<your_monitor_id>
   ```

### PipeWire/PulseAudio Reference Capture

**Problem:** Speaker output triggers ducking (false VAD detection)

**Cause:** On Linux with PipeWire, sounddevice's PortAudio backend cannot see monitor sources, causing AEC to be bypassed.

**Solution:** Use parec-based reference capture (automatically enabled by default):

1. Verify PulseAudio utilities are installed:
   ```bash
   which pactl parec
   # If not found: sudo apt install pulseaudio-utils
   ```

2. Check available monitor sources:
   ```bash
   pactl list short sources | grep monitor
   ```
   
   You should see entries like:
   - `alsa_output.*.monitor` - Built-in audio
   - `bluez_output.*.monitor` - Bluetooth speakers/headphones

3. Force parec method if auto-detection doesn't work:
   ```bash
   REF_CAPTURE_METHOD=parec python main.py
   ```

4. Expected output on successful detection:
   ```
   Reference: PipeWire monitor (bluez_output.*.monitor) via parec
   ```

**Note:** This is particularly important for Bluetooth speakers/headphones on Surface Pro and similar devices.

### SpeexDSP Build Issues

If `make` fails:

1. Ensure libspeexdsp-dev is installed
2. Check for pkg-config:
   ```bash
   pkg-config --exists speexdsp && echo "Found" || echo "Not found"
   ```

3. The system will automatically fall back to Python AEC if the build fails

### Volume Control Issues

If volume control doesn't work:

- Ensure `pactl` or `amixer` is available
- Check PulseAudio/PipeWire is running: `pactl info`
- Try setting volume manually: `pactl set-sink-volume @DEFAULT_SINK@ 50%`

## Differences from Original Code

### Fixed Issues

1. ✅ **Fixed bouncing volume during long conversations**
   - Old: Fixed 1.5s timer caused restore/re-duck cycles
   - New: Continuous speech state tracking

2. ✅ **Dynamic baseline tracking**
   - Old: User volume changes ignored during ducking
   - New: Detects and adapts to user volume changes

3. ✅ **Better AEC performance**
   - Old: Slow Python NLMS or basic C implementation
   - New: SpeexDSP with fallback

4. ✅ **Clean architecture**
   - Old: Monolithic files, inconsistent structure
   - New: Modular packages with clear interfaces

5. ✅ **Platform abstraction**
   - Old: Direct pactl/pycaw calls throughout code
   - New: Abstract VolumeController interface

### Architecture Improvements

- **Separation of concerns**: Each module has a single responsibility
- **Testability**: Clean interfaces make unit testing easy
- **Extensibility**: Plugin system for future enhancements
- **Cross-platform ready**: Abstract interfaces for easy porting

## Future Enhancements

The modular architecture makes it easy to add:

- [ ] Windows support (pycaw)
- [ ] macOS support (osascript)
- [ ] YamNET music classification plugin
- [ ] Multi-speaker detection
- [ ] Configurable ducking profiles
- [ ] Web-based GUI
- [ ] System tray integration
- [ ] Auto-start on boot

## Contributing

Contributions are welcome! The modular architecture makes it easy to add new features:

1. **Adding a new platform**: Implement `VolumeController` for your platform
2. **Adding a plugin**: Inherit from `AudioAnalysisPlugin`
3. **Improving AEC**: Swap in a different AEC implementation

## License

See the main repository LICENSE file.

## Acknowledgments

- **Silero Team** - For the excellent VAD model
- **SpeexDSP** - For robust echo cancellation
- **Original Conversation Mode** - For the initial concept and implementation
