# Conversation Mode V3 - Auto Ducking with Silero VAD

This folder contains the third version of the conversation mode implementation, featuring improved auto ducking with Silero VAD for voice activity detection.

## Features

- **Cross-Platform Support**: Works on both Windows and Linux systems
- **Real-time Voice Activity Detection**: Uses Silero VAD PyTorch model for accurate speech detection
- **Auto Ducking**: Automatically lowers system volume when speech is detected
- **Smart Volume Restoration**: Restores original volume after configurable delay
- **Thread-Safe**: Implements proper locking mechanisms for concurrent volume operations
- **Low Latency**: Processes audio with sub-50ms latency
- **Configurable Parameters**: Customizable ducking duration, percentage, and VAD threshold

## File Description

### auto_ducking_vad.py

The main implementation file that provides an `AudioDucker` class with the following capabilities:

- **AudioDucker Class**: Manages the entire auto ducking workflow
  - Initializes Silero VAD model
  - Handles platform-specific volume control (Windows/Linux)
  - Processes real-time audio streams
  - Manages ducking timers and state transitions

## Requirements

### Common Requirements
- Python 3.9 - 3.12
- torch
- numpy
- sounddevice

### Windows-Specific
- pycaw
- comtypes

### Linux-Specific
- PulseAudio or PipeWire with `pactl` command

## Installation

1. Install common dependencies:
```bash
pip install torch numpy sounddevice
```

2. For Windows:
```bash
pip install pycaw comtypes
```

3. For Linux:
Ensure PulseAudio or PipeWire is installed:
```bash
# Ubuntu/Debian
sudo apt-get install pulseaudio-utils

# Fedora
sudo dnf install pulseaudio-utils
```

## Usage

### Basic Usage

Run the script directly:

```bash
python auto_ducking_vad.py
```

### Advanced Usage

Import and customize the AudioDucker class:

```python
from auto_ducking_vad import AudioDucker

# Create a custom audio ducker
ducker = AudioDucker(
    sample_rate=16000,      # Audio sample rate in Hz
    frame_size=512,         # Samples per frame
    duck_duration=2.0,      # Keep volume ducked for 2 seconds
    duck_percentage=40,     # Duck to 40% of original volume
    vad_threshold=0.6       # Higher threshold = less sensitive
)

# Start the ducking system
ducker.run()
```

## Configuration Parameters

- **sample_rate** (default: 16000): Audio sampling rate in Hz. Silero VAD is optimized for 16kHz.
- **frame_size** (default: 512): Number of audio samples per processing frame.
- **duck_duration** (default: 1.5): Duration in seconds to keep volume lowered after speech detection.
- **duck_percentage** (default: 50): Percentage of original volume when ducked (0-100).
- **vad_threshold** (default: 0.5): VAD confidence threshold (0.0-1.0). Higher values = less sensitive.

## How It Works

1. **Audio Capture**: Captures audio from the default microphone at 16kHz
2. **VAD Processing**: Each audio frame is processed by Silero VAD to detect speech
3. **State Detection**: Monitors state transitions between SPEAKING and SILENCE
4. **Volume Ducking**: When speech is detected:
   - System volume is immediately lowered to the configured percentage
   - A timer starts to restore volume after the duck_duration
5. **Smart Timer**: If speech is detected again before the timer expires, the timer is reset
6. **Volume Restoration**: After silence is maintained for the duck_duration, volume is restored

## Architecture

The implementation follows a modular, object-oriented design:

- **Platform Abstraction**: Separate methods for Windows and Linux volume control
- **Thread Safety**: Uses locks to prevent race conditions in volume control
- **Timer Management**: Smart timer cancellation and restart for smooth transitions
- **Callback Architecture**: Efficient audio processing using sounddevice callbacks

## Performance

- **Latency**: Sub-50ms from speech detection to volume change
- **CPU Usage**: Minimal due to efficient PyTorch inference
- **Memory**: Low memory footprint with streaming audio processing

## Troubleshooting

### Windows Issues

**Problem**: "pycaw not available" warning
- **Solution**: Install pycaw and comtypes: `pip install pycaw comtypes`

### Linux Issues

**Problem**: "pactl not found" error
- **Solution**: Install PulseAudio utilities: `sudo apt-get install pulseaudio-utils`

**Problem**: Volume not changing
- **Solution**: Verify PulseAudio is running: `pactl info`

### General Issues

**Problem**: No audio detected
- **Solution**: Check microphone permissions and default input device

**Problem**: High CPU usage
- **Solution**: Increase frame_size to reduce processing frequency

## Contributing

This is part of the Conversation Mode project. For contributions and issues, please refer to the main repository.

## License

Same as the parent repository.

## Version History

- **v3.0** (Current): Initial release with cross-platform support and improved architecture
