#!/usr/bin/env python3
"""
Test the pw-record chase-read functionality.
"""

import sys
import os
import tempfile
import time
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

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

# Now import the module we want to test
from audio.capture import AudioCapture
import config


def test_pwrecord_preferred_over_parec():
    """Test that pw-record is preferred over parec when both are available."""
    capture = AudioCapture(
        mic_device_id=None,
        ref_device_id=None,
        ref_capture_method='auto'
    )
    
    # Mock both pw-record and parec as available
    def mock_which(cmd):
        if cmd == 'pw-record':
            return '/usr/bin/pw-record'
        elif cmd in ['parec', 'parecord']:
            return '/usr/bin/parecord'
        elif cmd == 'pactl':
            return '/usr/bin/pactl'
        return None
    
    # Mock pactl get-default-sink
    mock_get_sink = MagicMock()
    mock_get_sink.returncode = 0
    mock_get_sink.stdout = "test_sink"
    
    mock_popen = MagicMock()
    mock_popen.poll.return_value = None
    
    # Mock mic stream
    mock_mic_stream = MagicMock()
    mock_mic_stream.start = MagicMock()
    
    # Temporarily clear manual override
    old_value = config.REF_MONITOR_SOURCE
    config.REF_MONITOR_SOURCE = ""
    
    try:
        with patch('shutil.which', side_effect=mock_which):
            with patch('subprocess.run', return_value=mock_get_sink):
                with patch('subprocess.Popen', return_value=mock_popen) as mock_popen_call:
                    with patch('tempfile.mkstemp', return_value=(1, '/tmp/test.wav')):
                        with patch('os.close'):
                            with patch('time.sleep'):
                                with patch.object(MockSD, 'InputStream', return_value=mock_mic_stream):
                                    capture.start()
                                    
                                    # Verify pw-record was used
                                    assert capture.ref_process is not None, "ref_process should be set"
                                    assert capture.ref_temp_file == '/tmp/test.wav', "temp file should be set"
                                    
                                    # Verify the command used pw-record
                                    call_args = mock_popen_call.call_args[0][0]
                                    assert call_args[0] == 'pw-record', \
                                        f"Expected pw-record to be called, got: {call_args[0]}"
                                    print("✓ pw-record is preferred over parec")
                                    
                                    capture.stop()
    finally:
        config.REF_MONITOR_SOURCE = old_value


def test_parec_fallback_when_pwrecord_unavailable():
    """Test that parec is used as fallback when pw-record is not available."""
    capture = AudioCapture(
        mic_device_id=None,
        ref_device_id=None,
        ref_capture_method='auto'
    )
    
    # Mock only parec as available (pw-record not available)
    def mock_which(cmd):
        if cmd in ['parec', 'parecord']:
            return '/usr/bin/parecord'
        elif cmd == 'pactl':
            return '/usr/bin/pactl'
        return None  # pw-record not available
    
    # Mock pactl get-default-sink
    mock_get_sink = MagicMock()
    mock_get_sink.returncode = 0
    mock_get_sink.stdout = "test_sink"
    
    mock_popen = MagicMock()
    mock_popen.poll.return_value = None
    mock_popen.stdout = MagicMock()
    
    # Mock mic stream
    mock_mic_stream = MagicMock()
    mock_mic_stream.start = MagicMock()
    
    # Temporarily clear manual override
    old_value = config.REF_MONITOR_SOURCE
    config.REF_MONITOR_SOURCE = ""
    
    try:
        with patch('shutil.which', side_effect=mock_which):
            with patch('subprocess.run', return_value=mock_get_sink):
                with patch('subprocess.Popen', return_value=mock_popen) as mock_popen_call:
                    with patch.object(MockSD, 'InputStream', return_value=mock_mic_stream):
                        capture.start()
                        
                        # Verify parec was used as fallback
                        assert capture.ref_process is not None, "ref_process should be set"
                        assert capture.ref_temp_file is None, "temp file should not be set for parec"
                        
                        # Verify the command used parec
                        call_args = mock_popen_call.call_args[0][0]
                        assert call_args[0] in ['parecord', '/usr/bin/parecord'], \
                            f"Expected parec to be called, got: {call_args[0]}"
                        print("✓ parec is used as fallback when pw-record is unavailable")
                        
                        capture.stop()
    finally:
        config.REF_MONITOR_SOURCE = old_value


def test_pwrecord_cleanup_on_stop():
    """Test that temp file is cleaned up on stop."""
    capture = AudioCapture(
        mic_device_id=None,
        ref_device_id=None,
        ref_capture_method='auto'
    )
    
    def mock_which(cmd):
        if cmd == 'pw-record':
            return '/usr/bin/pw-record'
        elif cmd == 'pactl':
            return '/usr/bin/pactl'
        return None
    
    mock_get_sink = MagicMock()
    mock_get_sink.returncode = 0
    mock_get_sink.stdout = "test_sink"
    
    mock_popen = MagicMock()
    mock_popen.poll.return_value = None
    mock_popen.wait = MagicMock()
    
    # Mock mic stream
    mock_mic_stream = MagicMock()
    mock_mic_stream.start = MagicMock()
    mock_mic_stream.stop = MagicMock()
    mock_mic_stream.close = MagicMock()
    
    temp_file = '/tmp/test_cleanup.wav'
    
    # Temporarily clear manual override
    old_value = config.REF_MONITOR_SOURCE
    config.REF_MONITOR_SOURCE = ""
    
    try:
        with patch('shutil.which', side_effect=mock_which):
            with patch('subprocess.run', return_value=mock_get_sink):
                with patch('subprocess.Popen', return_value=mock_popen):
                    with patch('tempfile.mkstemp', return_value=(1, temp_file)):
                        with patch('os.close'):
                            with patch('time.sleep'):
                                with patch.object(MockSD, 'InputStream', return_value=mock_mic_stream):
                                    with patch('os.path.exists', return_value=True):
                                        with patch('os.unlink') as mock_unlink:
                                            capture.start()
                                            assert capture.ref_temp_file == temp_file
                                            
                                            capture.stop()
                                            
                                            # Verify temp file was deleted
                                            mock_unlink.assert_called_once_with(temp_file)
                                            assert capture.ref_temp_file is None
                                            print("✓ Temp file is cleaned up on stop")
    finally:
        config.REF_MONITOR_SOURCE = old_value


def test_pwrecord_manual_override_priority():
    """Test that manual REF_MONITOR_SOURCE override is used with pw-record."""
    # Set manual override
    old_value = config.REF_MONITOR_SOURCE
    config.REF_MONITOR_SOURCE = "manual.monitor.source"
    
    try:
        capture = AudioCapture(
            mic_device_id=None,
            ref_device_id=None,
            ref_capture_method='auto'
        )
        
        def mock_which(cmd):
            if cmd == 'pw-record':
                return '/usr/bin/pw-record'
            return None
        
        mock_popen = MagicMock()
        mock_popen.poll.return_value = None
        
        # Mock mic stream
        mock_mic_stream = MagicMock()
        mock_mic_stream.start = MagicMock()
        
        with patch('shutil.which', side_effect=mock_which):
            with patch('subprocess.run') as mock_run:
                with patch('subprocess.Popen', return_value=mock_popen) as mock_popen_call:
                    with patch('tempfile.mkstemp', return_value=(1, '/tmp/test.wav')):
                        with patch('os.close'):
                            with patch('time.sleep'):
                                with patch.object(MockSD, 'InputStream', return_value=mock_mic_stream):
                                    capture.start()
                                    
                                    # Verify manual override was used
                                    assert capture.ref_monitor_source == "manual.monitor.source"
                                    
                                    # Verify pactl was NOT called (manual override bypasses detection)
                                    assert not mock_run.called, "pactl should not be called with manual override"
                                    
                                    # Verify pw-record was called with the manual override
                                    call_args = mock_popen_call.call_args[0][0]
                                    assert '--target=manual.monitor.source' in call_args, \
                                        f"Expected manual override in command, got: {call_args}"
                                    print("✓ Manual override has priority with pw-record")
                                    
                                    capture.stop()
    finally:
        config.REF_MONITOR_SOURCE = old_value


if __name__ == "__main__":
    print("=" * 60)
    print("Testing pw-record chase-read functionality")
    print("=" * 60)
    print()
    
    try:
        test_pwrecord_preferred_over_parec()
        test_parec_fallback_when_pwrecord_unavailable()
        test_pwrecord_cleanup_on_stop()
        test_pwrecord_manual_override_priority()
        
        print()
        print("=" * 60)
        print("All pw-record tests passed! ✓")
        print("=" * 60)
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        sys.exit(1)
