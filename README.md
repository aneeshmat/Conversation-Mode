# Conversation-Mode

Real‑time voice‑activity‑driven system audio ducking for Windows.

Conversation Mode is a Python-based real‑time audio processing tool that automatically lowers (“ducks”) system volume when human speech is detected. It is built on PyTorch, Silero VAD, sounddevice, and PyCaw, and is engineered for low-latency (<50ms) responsiveness suitable for voice assistants, streaming, meetings, and accessibility use cases.

# Features
Real‑time Voice Activity Detection
Utilizes the Silero VAD PyTorch model to classify speech frames with sub‑50ms latency.

System Audio Ducking (Windows) - 
Integrates with Windows’ audio stack through PyCaw and COM interfaces to automatically lower and restore system volume.

Thread‑Safe Volume Restoration - 
Uses synchronized timers, locks, and state management to ensure consistent volume transitions even during rapid speech/silence switches.

Continuous Microphone Stream Processing - 
Processes live audio using sounddevice with a low‑latency callback architecture (512‑sample frames @ 16kHz).

Modular Architecture - 
Designed to support future enhancements such as:
Normalized Least Means Squares (NLMS) acoustic echo cancellation, 
Google YAMNet‑based audio/voice/singing classification, 
Statistical RMS‑based noise gating or activity detection

# How It Works
Conversation Mode monitors your microphone in real time and performs the following actions:
Capture audio frames at 16kHz (sounddevice.InputStream), 
Run VAD inference using Silero (PyTorch) on each frame.

When human speech is detected:
System audio volume is ducked (typically to 50%); 
a smart timer manages how long volume stays ducked.

After speech ends:
The original system volume is restored automatically; 
volume changes are thread‑safe, preventing overlapping operations.


# Requirements

OS: Windows 10/11

Python: 3.9–3.12

Packages:
torch, numpy, sounddevice, pycaw, comtypes
