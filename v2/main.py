"""
Conversation Mode v2 - Main Entry Point

Real-time voice-activity-driven audio ducking with:
- Smart auto-ducking with dynamic baseline tracking
- Acoustic Echo Cancellation using SpeexDSP
- Voice Activity Detection using Silero VAD
- Cross-platform volume control
- Clean modular architecture

Usage:
    python main.py

Environment Variables:
    MIC_DEVICE_ID - Microphone device ID (-1 for default)
    REF_DEVICE_ID - Reference/monitor device ID (-1 for default)
    DEBUG - Enable debug logging (0 or 1)
"""

import sys
from audio import AudioCapture, get_volume_controller
from pipeline import ConversationWorker
from gui import run_gui
import config


def main():
    """Main entry point."""
    print("=" * 60)
    print("Conversation Mode v2")
    print("=" * 60)
    print()
    
    # Get volume controller
    print("Initializing volume controller...")
    volume_controller = get_volume_controller()
    
    if volume_controller is None:
        print("ERROR: Could not initialize volume controller for this platform.")
        print("Currently supported: Linux (pactl/amixer)")
        return 1
    
    print(f"✓ Volume controller initialized")
    print()
    
    # Initialize audio capture
    print("Initializing audio capture...")
    audio_capture = AudioCapture(
        mic_device_id=config.MIC_DEVICE_ID,
        ref_device_id=config.REF_DEVICE_ID,
        ref_capture_method=config.REF_CAPTURE_METHOD
    )
    
    device_info = audio_capture.get_device_info()
    print(f"  Microphone: {device_info['mic_device_name']}")
    print(f"  Reference:  {device_info['ref_device_name']}")
    
    # Warn if no reference source is available
    if device_info['ref_device_name'] == 'Not configured':
        print()
        print("WARNING: No reference source available. Speaker output may trigger ducking.")
        print("  To fix this on Linux with PipeWire/PulseAudio:")
        print("    1. Set REF_CAPTURE_METHOD=parec to enable monitor source detection")
        print("    2. Set REF_MONITOR_SOURCE=<source_name> to manually specify the monitor")
        print("       (find source names with: pactl list short sources)")
        print("    3. Or set REF_DEVICE_ID to a valid device ID for sounddevice")
        print()
    
    print(f"  Sample Rate: {device_info['sample_rate']} Hz")
    print(f"  Frame Size:  {device_info['frame_size']} samples")
    print()
    
    # Initialize worker
    print("Initializing processing pipeline...")
    worker = ConversationWorker(audio_capture, volume_controller)
    
    aec_status = worker.get_aec_status()
    print(f"  AEC: {aec_status.value}")
    print(f"  VAD: Silero")
    print()
    
    # Check if SpeexDSP AEC is available
    if aec_status.name == "FALLBACK_ACTIVE":
        print("WARNING: SpeexDSP not available, using fallback AEC")
        print("  To build SpeexDSP library:")
        print("    1. Install libspeexdsp-dev: sudo apt install libspeexdsp-dev")
        print("    2. Run: make")
        print()
    elif aec_status.name == "DISABLED":
        print("INFO: AEC is disabled")
        print()
    
    # Start GUI
    print("Starting GUI...")
    print("=" * 60)
    print()
    
    try:
        run_gui(worker, audio_capture, volume_controller)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        worker.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
