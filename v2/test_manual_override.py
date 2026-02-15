#!/usr/bin/env python3
"""
Test the manual REF_MONITOR_SOURCE override functionality.
"""

import sys
import os
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

# Set the manual override before importing config
os.environ['REF_MONITOR_SOURCE'] = 'manually.specified.monitor'

# Now import modules
import config
from audio.capture import AudioCapture

def test_manual_override():
    """Test that REF_MONITOR_SOURCE from config is used when set."""
    # Verify config loaded the environment variable
    assert config.REF_MONITOR_SOURCE == 'manually.specified.monitor', \
        f"Config should have loaded REF_MONITOR_SOURCE, got: {config.REF_MONITOR_SOURCE}"
    print(f"✓ Config loaded REF_MONITOR_SOURCE: {config.REF_MONITOR_SOURCE}")
    
    # Create AudioCapture instance with parec method
    capture = AudioCapture(
        mic_device_id=None,
        ref_device_id=None,
        ref_capture_method='parec'
    )
    
    # Mock shutil.which to return a valid parec path
    # Mock subprocess.Popen to avoid actually running parec
    mock_popen = MagicMock()
    mock_popen.poll.return_value = None
    mock_popen.stdout = MagicMock()
    
    with patch('shutil.which', return_value='/usr/bin/parecord'):
        with patch('subprocess.Popen', return_value=mock_popen) as mock_popen_call:
            with patch('subprocess.run') as mock_run:
                # This should not be called since we have manual override
                mock_run.return_value = MagicMock(returncode=0, stdout="")
                
                # Start capture (this will call _start_parec_ref)
                capture.start()
                
                # Check that the manual monitor source was used
                assert capture.ref_monitor_source == 'manually.specified.monitor', \
                    f"Expected manual override to be used, got: {capture.ref_monitor_source}"
                print(f"✓ Manual override used: {capture.ref_monitor_source}")
                
                # Verify that pactl was NOT called (since we used manual override)
                assert not mock_run.called, "pactl should not be called when manual override is set"
                print("✓ Auto-detection bypassed (pactl not called)")
                
                # Verify Popen was called with the correct device
                mock_popen_call.assert_called_once()
                call_args = mock_popen_call.call_args[0][0]
                # Check for either --device= (parec) or --target= (pw-record)
                assert ('--device=manually.specified.monitor' in call_args or 
                        '--target=manually.specified.monitor' in call_args), \
                    f"Expected manual monitor in command, got: {call_args}"
                print(f"✓ Capture called with correct device: {capture.ref_monitor_source}")
                
                capture.stop()


def test_empty_override_uses_autodetect():
    """Test that empty REF_MONITOR_SOURCE falls back to auto-detection."""
    # Temporarily clear the override
    old_value = config.REF_MONITOR_SOURCE
    config.REF_MONITOR_SOURCE = ""
    
    try:
        capture = AudioCapture(
            mic_device_id=None,
            ref_device_id=None,
            ref_capture_method='parec'
        )
        
        # Mock for pactl get-default-sink (returns just the sink name)
        mock_get_sink_result = MagicMock()
        mock_get_sink_result.returncode = 0
        mock_get_sink_result.stdout = "bluez_output.test"
        
        mock_popen = MagicMock()
        mock_popen.poll.return_value = None
        mock_popen.stdout = MagicMock()
        
        with patch('shutil.which', return_value='/usr/bin/parecord'):
            with patch('subprocess.run', return_value=mock_get_sink_result) as mock_run:
                with patch('subprocess.Popen', return_value=mock_popen):
                    capture.start()
                    
                    # Verify pactl was called for auto-detection
                    assert mock_run.called, "pactl should be called when no manual override"
                    print("✓ Auto-detection used when REF_MONITOR_SOURCE is empty")
                    
                    # Verify the auto-detected source was used
                    assert capture.ref_monitor_source == 'bluez_output.test.monitor', \
                        f"Expected auto-detected monitor, got: {capture.ref_monitor_source}"
                    print(f"✓ Auto-detected monitor used: {capture.ref_monitor_source}")
                    
                    capture.stop()
    finally:
        # Restore original value
        config.REF_MONITOR_SOURCE = old_value


if __name__ == "__main__":
    print("=" * 60)
    print("Testing REF_MONITOR_SOURCE manual override")
    print("=" * 60)
    print()
    
    try:
        test_manual_override()
        print()
        test_empty_override_uses_autodetect()
        
        print()
        print("=" * 60)
        print("All manual override tests passed! ✓")
        print("=" * 60)
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        sys.exit(1)
