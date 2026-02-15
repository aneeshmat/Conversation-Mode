"""
Smart auto-ducking controller with dynamic baseline tracking.
Handles volume reduction when speech is detected and restoration when speech ends.
"""

import time
import numpy as np
from typing import Callable

# Handle both package and direct execution
try:
    from .. import config
except ImportError:
    import config


class DuckController:
    """
    Smart ducking controller that:
    1. Lowers volume when speech is detected
    2. Holds ducking while speech continues
    3. Restores volume after speech ends (with hold period)
    4. Tracks dynamic baseline (detects user volume changes during ducking)
    """
    
    def __init__(self, get_volume_fn: Callable[[], int], 
                 set_volume_fn: Callable[[int], bool]):
        """
        Initialize ducking controller.
        
        Args:
            get_volume_fn: Function to get current system volume (0-100)
            set_volume_fn: Function to set system volume (0-100)
        """
        self.get_volume = get_volume_fn
        self.set_volume = set_volume_fn
        
        # Configuration
        self.duck_ratio = config.DUCK_RATIO
        self.min_volume = config.DUCK_MIN_PERCENT
        self.hold_duration = config.DUCK_HOLD_SEC
        self.smooth_steps = config.DUCK_SMOOTH_STEPS
        self.smooth_step_ms = config.DUCK_SMOOTH_STEP_MS
        self.baseline_threshold = config.DUCK_BASELINE_CHANGE_THRESHOLD
        
        # State
        self.ducked = False
        self.baseline_volume = None
        self.expected_ducked_volume = None
        self.last_speech_time = 0.0
    
    def _smooth_transition(self, from_vol: int, to_vol: int):
        """
        Smoothly transition from one volume to another.
        
        Args:
            from_vol: Starting volume
            to_vol: Target volume
        """
        if self.smooth_steps <= 1 or from_vol == to_vol:
            self.set_volume(to_vol)
            return
        
        steps = np.linspace(from_vol, to_vol, self.smooth_steps)
        for step_vol in steps:
            self.set_volume(int(round(step_vol)))
            time.sleep(self.smooth_step_ms / 1000.0)
    
    def notify_speech(self):
        """
        Notify controller that speech was detected.
        Called on every frame where speech is active.
        """
        current_time = time.monotonic()
        self.last_speech_time = current_time
        
        if not self.ducked:
            # Not currently ducked - start ducking
            current_vol = self.get_volume()
            if current_vol < 0:
                return  # Can't get volume
            
            # Calculate target ducked volume
            target_vol = max(
                int(round(current_vol * self.duck_ratio)),
                self.min_volume
            )
            
            # Smooth transition to ducked volume
            self._smooth_transition(current_vol, target_vol)
            
            # Update state
            self.baseline_volume = current_vol
            self.expected_ducked_volume = target_vol
            self.ducked = True
            
            if config.DEBUG:
                print(f"[Duck] Started: {current_vol}% -> {target_vol}% (baseline: {self.baseline_volume}%)")
        
        else:
            # Already ducked - check if user changed volume
            current_vol = self.get_volume()
            if current_vol < 0:
                return
            
            # Check if current volume differs significantly from expected ducked volume
            diff = abs(current_vol - self.expected_ducked_volume)
            
            if diff > self.baseline_threshold:
                # User changed volume! Update baseline
                if config.DEBUG:
                    print(f"[Duck] User changed volume: {self.expected_ducked_volume}% -> {current_vol}%")
                
                # Current volume IS the new baseline (user's intent)
                new_baseline = current_vol
                
                # Calculate new ducked target
                new_target = max(
                    int(round(new_baseline * self.duck_ratio)),
                    self.min_volume
                )
                
                # Smooth transition to new target
                self._smooth_transition(current_vol, new_target)
                
                # Update state
                self.baseline_volume = new_baseline
                self.expected_ducked_volume = new_target
                
                if config.DEBUG:
                    print(f"[Duck] New baseline: {new_baseline}% (target: {new_target}%)")
    
    def update(self):
        """
        Update ducking state.
        Should be called periodically (e.g., on every audio frame).
        Handles restoration when speech ends.
        """
        if not self.ducked:
            return
        
        current_time = time.monotonic()
        time_since_speech = current_time - self.last_speech_time
        
        # Check if hold period has elapsed
        if time_since_speech >= self.hold_duration:
            # Restore to baseline
            current_vol = self.get_volume()
            
            if current_vol >= 0 and self.baseline_volume is not None:
                # Only restore if we're still at (or near) the expected ducked volume
                # (if user changed volume, don't mess with it)
                diff = abs(current_vol - self.expected_ducked_volume)
                
                if diff <= self.baseline_threshold:
                    # Safe to restore
                    self._smooth_transition(current_vol, self.baseline_volume)
                    
                    if config.DEBUG:
                        print(f"[Duck] Restored: {current_vol}% -> {self.baseline_volume}%")
                else:
                    if config.DEBUG:
                        print(f"[Duck] Skipped restore (user changed volume)")
            
            # Clear ducking state
            self.ducked = False
            self.baseline_volume = None
            self.expected_ducked_volume = None
    
    def is_ducked(self) -> bool:
        """Check if currently ducked."""
        return self.ducked
    
    def get_baseline(self) -> int:
        """Get baseline volume, or -1 if not ducked."""
        return self.baseline_volume if self.baseline_volume is not None else -1
    
    def force_restore(self):
        """
        Force immediate restoration (for shutdown/stop).
        """
        if self.ducked and self.baseline_volume is not None:
            current_vol = self.get_volume()
            
            if current_vol >= 0:
                # Check if we should restore
                diff = abs(current_vol - self.expected_ducked_volume)
                
                if diff <= self.baseline_threshold:
                    self._smooth_transition(current_vol, self.baseline_volume)
                    
                    if config.DEBUG:
                        print(f"[Duck] Force restored: {current_vol}% -> {self.baseline_volume}%")
        
        # Clear state
        self.ducked = False
        self.baseline_volume = None
        self.expected_ducked_volume = None
    
    def reset(self):
        """Reset controller state without changing volume."""
        self.ducked = False
        self.baseline_volume = None
        self.expected_ducked_volume = None
        self.last_speech_time = 0.0
