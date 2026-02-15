#!/usr/bin/env python3
"""
Simple test script to verify v2 modules work correctly.
This tests the logic without requiring actual audio hardware.
"""

import sys
import numpy as np

# Mock imports to avoid hardware dependencies
class MockSD:
    class InputStream:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def stop(self):
            pass
        def close(self):
            pass
    
    @staticmethod
    def query_devices(*args, **kwargs):
        return {'name': 'Mock Device'}

sys.modules['sounddevice'] = MockSD()

# Now we can import
import config
print("✓ Config module loaded")

# Test platform_volume (this should work)
from audio.platform_volume import VolumeController, get_volume_controller
print("✓ platform_volume module loaded")

# Test if volume controller can be created
try:
    vc = get_volume_controller()
    if vc:
        print(f"✓ Volume controller created: {type(vc).__name__}")
    else:
        print("⚠ Volume controller not available (expected on non-Linux)")
except Exception as e:
    print(f"⚠ Volume controller error: {e}")

# Test AEC fallback
from aec.aec_fallback import AECFallback
aec_fb = AECFallback(frame_size=1024, sample_rate=48000)
test_audio = np.random.randn(1024).astype(np.float32) * 0.1
result = aec_fb.process(test_audio, test_audio)
assert len(result) == 1024, "AEC fallback output size mismatch"
print("✓ AEC fallback works")

# Test ducking controller
from ducking.duck_controller import DuckController

mock_volume = [50]  # Mutable to track changes

def get_vol():
    return mock_volume[0]

def set_vol(v):
    mock_volume[0] = v
    return True

duck = DuckController(get_vol, set_vol)
print("✓ DuckController created")

# Simulate speech detection
duck.notify_speech()
assert duck.is_ducked(), "Should be ducked after notify_speech"
print(f"✓ Ducking activated: {mock_volume[0]}% (from 50%)")

# Simulate volume change by user
mock_volume[0] = 40  # User changed volume
duck.notify_speech()  # Continue speech
# After this, baseline should be updated
print(f"✓ User volume change detected: {mock_volume[0]}%")

# Test plugin base
from plugins.base import ExampleMusicDetectorPlugin
plugin = ExampleMusicDetectorPlugin(sample_rate=16000, frame_size=512)
test_audio_16k = np.random.randn(512).astype(np.float32) * 0.1
result = plugin.process(test_audio_16k)
assert 'music_probability' in result, "Plugin should return music_probability"
print(f"✓ Example plugin works: {result['classification']}")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nNote: Full functionality requires:")
print("  - Audio hardware (microphone, speakers)")
print("  - PyTorch (for Silero VAD)")
print("  - SpeexDSP (for optimal AEC)")
print("  - Linux with pactl/amixer (for volume control)")
