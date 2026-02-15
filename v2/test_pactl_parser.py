#!/usr/bin/env python3
"""
Test the PipeWire monitor source detection parser.
Specifically tests the fix for tab-delimited pactl output.
"""

import sys
import subprocess
from unittest.mock import patch, MagicMock

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

# Now import the function we want to test
from audio.capture import _detect_pipewire_monitor_source

# Sample pactl output with problematic formatting
# Line 324 has concatenated fields (no tab between source name and driver)
PROBLEMATIC_PACTL_OUTPUT = """239	alsa_output.pci-0000_00_1f.3.analog-stereo.monitor	PipeWire	s32le 2ch 48000Hz	SUSPENDED
240	alsa_input.pci-0000_00_1f.3.analog-stereo	PipeWire	s32le 2ch 48000Hz	SUSPENDED
324	alsa_output.usb-16042020V2_JLAB_TALK_MICROPHONE-00.analog-stereo.monitorPipeWire	s24le 2ch 48000Hz	SUSPENDED
325	alsa_input.usb-16042020V2_JLAB_TALK_MICROPHONE-00.analog-stereo	PipeWires24le 2ch 48000Hz	SUSPENDED
365	bluez_output.20_64_DE_A6_AA_69.1.monitor	PipeWire	s16le 2ch 48000Hz	RUNNING"""

# Normal tab-delimited pactl output
NORMAL_PACTL_OUTPUT = """239	alsa_output.pci-0000_00_1f.3.analog-stereo.monitor	PipeWire	s32le 2ch 48000Hz	SUSPENDED
240	alsa_input.pci-0000_00_1f.3.analog-stereo	PipeWire	s32le 2ch 48000Hz	SUSPENDED
365	bluez_output.20_64_DE_A6_AA_69.1.monitor	PipeWire	s16le 2ch 48000Hz	RUNNING"""

# Output with only suspended monitors
SUSPENDED_ONLY_OUTPUT = """239	alsa_output.pci-0000_00_1f.3.analog-stereo.monitor	PipeWire	s32le 2ch 48000Hz	SUSPENDED
240	alsa_input.pci-0000_00_1f.3.analog-stereo	PipeWire	s32le 2ch 48000Hz	SUSPENDED"""

# Empty output
EMPTY_OUTPUT = ""


def test_problematic_output():
    """Test parsing of problematic pactl output with concatenated fields."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = PROBLEMATIC_PACTL_OUTPUT
    
    with patch('subprocess.run', return_value=mock_result):
        with patch('shutil.which', return_value='/usr/bin/pactl'):
            result = _detect_pipewire_monitor_source()
            assert result == 'bluez_output.20_64_DE_A6_AA_69.1.monitor', \
                f"Expected running monitor, got: {result}"
            print("✓ Problematic output parsed correctly (found RUNNING monitor)")


def test_normal_output():
    """Test parsing of normal tab-delimited pactl output."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = NORMAL_PACTL_OUTPUT
    
    with patch('subprocess.run', return_value=mock_result):
        with patch('shutil.which', return_value='/usr/bin/pactl'):
            result = _detect_pipewire_monitor_source()
            assert result == 'bluez_output.20_64_DE_A6_AA_69.1.monitor', \
                f"Expected running monitor, got: {result}"
            print("✓ Normal output parsed correctly (found RUNNING monitor)")


def test_suspended_only():
    """Test fallback to suspended monitor when no running monitor exists."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = SUSPENDED_ONLY_OUTPUT
    
    with patch('subprocess.run', return_value=mock_result):
        with patch('shutil.which', return_value='/usr/bin/pactl'):
            result = _detect_pipewire_monitor_source()
            assert result == 'alsa_output.pci-0000_00_1f.3.analog-stereo.monitor', \
                f"Expected first monitor as fallback, got: {result}"
            print("✓ Suspended-only output handled correctly (used fallback)")


def test_empty_output():
    """Test handling of empty pactl output."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = EMPTY_OUTPUT
    
    with patch('subprocess.run', return_value=mock_result):
        with patch('shutil.which', return_value='/usr/bin/pactl'):
            result = _detect_pipewire_monitor_source()
            assert result is None, f"Expected None for empty output, got: {result}"
            print("✓ Empty output handled correctly (returned None)")


def test_pactl_not_available():
    """Test handling when pactl is not installed."""
    with patch('shutil.which', return_value=None):
        result = _detect_pipewire_monitor_source()
        assert result is None, f"Expected None when pactl unavailable, got: {result}"
        print("✓ Missing pactl handled correctly (returned None)")


def test_pactl_error():
    """Test handling when pactl command fails."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    
    with patch('subprocess.run', return_value=mock_result):
        with patch('shutil.which', return_value='/usr/bin/pactl'):
            result = _detect_pipewire_monitor_source()
            assert result is None, f"Expected None on command error, got: {result}"
            print("✓ Command error handled correctly (returned None)")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing PipeWire monitor source detection parser")
    print("=" * 60)
    print()
    
    try:
        test_problematic_output()
        test_normal_output()
        test_suspended_only()
        test_empty_output()
        test_pactl_not_available()
        test_pactl_error()
        
        print()
        print("=" * 60)
        print("All parser tests passed! ✓")
        print("=" * 60)
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        sys.exit(1)
