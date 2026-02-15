"""
Configuration file for Conversation Mode v2.
All tunable parameters in one place.
"""

import os

# ============================================================================
# Audio Configuration
# ============================================================================
SAMPLE_RATE = 16000  # VAD processing rate (Hz)
DEVICE_RATE = 48000  # Hardware I/O rate (Hz)
FRAME_SIZE = 1024    # Samples per audio frame at device rate

# Audio device IDs (from environment variables with sensible defaults)
MIC_DEVICE_ID = int(os.getenv("MIC_DEVICE_ID", 5))
REF_DEVICE_ID = int(os.getenv("REF_DEVICE_ID", 12))

# Reference capture method: 'auto', 'sounddevice', 'parec', 'none'
# 'auto' - Use sounddevice if REF_DEVICE_ID is set, otherwise try parec for PipeWire/PulseAudio monitor
# 'sounddevice' - Force sounddevice (requires valid REF_DEVICE_ID)
# 'parec' - Force parec subprocess for PipeWire/PulseAudio monitor
# 'none' - Disable reference capture (no AEC)
REF_CAPTURE_METHOD = os.getenv("REF_CAPTURE_METHOD", "auto")

# Manual override for PipeWire/PulseAudio monitor source name
# Set this if auto-detection fails (e.g., "bluez_output.20_64_DE_A6_AA_69.1.monitor")
# Find source names with: pactl list short sources
REF_MONITOR_SOURCE = os.getenv("REF_MONITOR_SOURCE", "")

# ============================================================================
# Ducking Configuration
# ============================================================================
DUCK_RATIO = 0.5              # Duck to 50% of baseline volume
DUCK_MIN_PERCENT = 5          # Minimum ducked volume (%)
DUCK_HOLD_SEC = 1.5           # Hold ducking after speech stops (seconds)
DUCK_SMOOTH_STEPS = 8         # Number of interpolation steps for smooth transitions
DUCK_SMOOTH_STEP_MS = 25      # Milliseconds between smooth volume steps
DUCK_BASELINE_CHANGE_THRESHOLD = 3  # Percent difference to detect user volume change

# ============================================================================
# Voice Activity Detection (VAD) Configuration
# ============================================================================
VAD_ON_THRESHOLD = 0.35       # EMA probability to trigger speech
VAD_OFF_THRESHOLD = 0.20      # EMA probability to end speech
VAD_EMA_ALPHA = 0.5           # EMA smoothing factor (0-1, higher = more responsive)
VAD_HOLD_FRAMES = 10          # Hold frames after speech probability drops

# ============================================================================
# Acoustic Echo Cancellation (AEC) Configuration
# ============================================================================
AEC_FILTER_LENGTH = 1024      # SpeexDSP filter length in samples
AEC_DELAY = 0                 # Estimated delay between reference and mic (0 = auto)
AEC_ENABLED_DEFAULT = True    # Enable AEC by default

# ============================================================================
# Audio Processing Configuration
# ============================================================================
HP_ALPHA = 0.995              # DC-blocking high-pass filter coefficient
GAIN_AFTER_AEC = 2.0          # Gain boost after echo cancellation

# ============================================================================
# Debug Configuration
# ============================================================================
DEBUG = os.getenv("DEBUG", "0") == "1"  # Enable debug logging

# ============================================================================
# GUI Configuration
# ============================================================================
GUI_UPDATE_MS = 50            # GUI update interval in milliseconds
