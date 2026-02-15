"""
Tkinter GUI for Conversation Mode v2.
Displays real-time status and controls.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional

# Handle both package and direct execution
try:
    from ..pipeline import ConversationWorker
    from ..audio import AudioCapture, VolumeController
    from .. import config
except ImportError:
    from pipeline import ConversationWorker
    from audio import AudioCapture, VolumeController
    import config


class ConversationGUI:
    """
    Simple Tkinter GUI for Conversation Mode.
    """
    
    def __init__(self, root: tk.Tk, worker: ConversationWorker, 
                 audio: AudioCapture, volume: VolumeController):
        """
        Initialize GUI.
        
        Args:
            root: Tkinter root window
            worker: Conversation worker instance
            audio: Audio capture instance
            volume: Volume controller instance
        """
        self.root = root
        self.worker = worker
        self.audio = audio
        self.volume = volume
        
        self.root.title("Conversation Mode v2")
        self.root.geometry("500x450")
        self.root.resizable(False, False)
        
        # State
        self.aec_enabled = config.AEC_ENABLED_DEFAULT
        
        self._setup_ui()
        self._start_update_loop()
    
    def _setup_ui(self):
        """Setup UI elements."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = ttk.Label(main_frame, text="Conversation Mode v2", 
                         font=("Arial", 16, "bold"))
        title.pack(pady=(0, 10))
        
        # Control Frame
        control_frame = ttk.LabelFrame(main_frame, text="Control", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Start/Stop button
        self.start_stop_btn = ttk.Button(
            control_frame, 
            text="Start", 
            command=self._toggle_start_stop,
            width=15
        )
        self.start_stop_btn.pack()
        
        # AEC Toggle
        aec_frame = ttk.Frame(control_frame)
        aec_frame.pack(pady=(10, 0))
        
        self.aec_var = tk.BooleanVar(value=self.aec_enabled)
        self.aec_checkbox = ttk.Checkbutton(
            aec_frame,
            text="Enable Acoustic Echo Cancellation (AEC)",
            variable=self.aec_var,
            command=self._toggle_aec
        )
        self.aec_checkbox.pack()
        
        # Status Frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # VAD Probability
        vad_frame = ttk.Frame(status_frame)
        vad_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(vad_frame, text="VAD Probability:", width=20).pack(side=tk.LEFT)
        self.vad_label = ttk.Label(vad_frame, text="0.00", font=("Arial", 10, "bold"))
        self.vad_label.pack(side=tk.LEFT)
        
        self.vad_progress = ttk.Progressbar(vad_frame, length=200, mode='determinate')
        self.vad_progress.pack(side=tk.LEFT, padx=(10, 0))
        
        # Speech State
        speech_frame = ttk.Frame(status_frame)
        speech_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(speech_frame, text="Speech:", width=20).pack(side=tk.LEFT)
        self.speech_label = ttk.Label(speech_frame, text="Inactive", 
                                      font=("Arial", 10, "bold"))
        self.speech_label.pack(side=tk.LEFT)
        
        # Ducking State
        duck_frame = ttk.Frame(status_frame)
        duck_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(duck_frame, text="Ducking:", width=20).pack(side=tk.LEFT)
        self.duck_label = ttk.Label(duck_frame, text="Inactive", 
                                    font=("Arial", 10, "bold"))
        self.duck_label.pack(side=tk.LEFT)
        
        # Current Volume
        vol_frame = ttk.Frame(status_frame)
        vol_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(vol_frame, text="Current Volume:", width=20).pack(side=tk.LEFT)
        self.vol_label = ttk.Label(vol_frame, text="--", font=("Arial", 10, "bold"))
        self.vol_label.pack(side=tk.LEFT)
        
        # Baseline Volume
        baseline_frame = ttk.Frame(status_frame)
        baseline_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(baseline_frame, text="Baseline Volume:", width=20).pack(side=tk.LEFT)
        self.baseline_label = ttk.Label(baseline_frame, text="--", 
                                        font=("Arial", 10, "bold"))
        self.baseline_label.pack(side=tk.LEFT)
        
        # AEC Status
        aec_status_frame = ttk.Frame(status_frame)
        aec_status_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(aec_status_frame, text="AEC Status:", width=20).pack(side=tk.LEFT)
        self.aec_status_label = ttk.Label(aec_status_frame, text="Disabled", 
                                          font=("Arial", 10, "bold"))
        self.aec_status_label.pack(side=tk.LEFT)
        
        # Device Info Frame
        device_frame = ttk.LabelFrame(main_frame, text="Audio Devices", padding="10")
        device_frame.pack(fill=tk.X)
        
        device_info = self.audio.get_device_info()
        
        # Microphone
        mic_frame = ttk.Frame(device_frame)
        mic_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(mic_frame, text="Microphone:", width=12).pack(side=tk.LEFT)
        ttk.Label(mic_frame, text=device_info['mic_device_name'], 
                 font=("Arial", 9)).pack(side=tk.LEFT)
        
        # Reference
        ref_frame = ttk.Frame(device_frame)
        ref_frame.pack(fill=tk.X)
        ttk.Label(ref_frame, text="Reference:", width=12).pack(side=tk.LEFT)
        ttk.Label(ref_frame, text=device_info['ref_device_name'], 
                 font=("Arial", 9)).pack(side=tk.LEFT)
    
    def _toggle_start_stop(self):
        """Handle start/stop button click."""
        if self.worker.is_running():
            self.worker.stop()
            self.start_stop_btn.config(text="Start")
        else:
            success = self.worker.start()
            if success:
                self.start_stop_btn.config(text="Stop")
            else:
                # Show error
                error_label = ttk.Label(self.root, text="Failed to start audio capture!", 
                                       foreground="red")
                error_label.pack()
                self.root.after(3000, error_label.destroy)
    
    def _toggle_aec(self):
        """Handle AEC checkbox toggle."""
        enabled = self.aec_var.get()
        self.worker.set_aec_enabled(enabled)
    
    def _update_status(self):
        """Update status displays."""
        # VAD probability
        vad_prob = self.worker.get_vad_probability()
        self.vad_label.config(text=f"{vad_prob:.2f}")
        self.vad_progress['value'] = vad_prob * 100
        
        # Speech state
        speech_active = self.worker.is_speech_active()
        if speech_active:
            self.speech_label.config(text="Active", foreground="green")
        else:
            self.speech_label.config(text="Inactive", foreground="gray")
        
        # Ducking state
        ducked = self.worker.is_ducked()
        if ducked:
            self.duck_label.config(text="Active", foreground="orange")
        else:
            self.duck_label.config(text="Inactive", foreground="gray")
        
        # Current volume
        current_vol = self.volume.get_volume()
        if current_vol >= 0:
            self.vol_label.config(text=f"{current_vol}%")
        else:
            self.vol_label.config(text="--")
        
        # Baseline volume
        baseline_vol = self.worker.get_baseline_volume()
        if baseline_vol >= 0:
            self.baseline_label.config(text=f"{baseline_vol}%")
        else:
            self.baseline_label.config(text="--")
        
        # AEC status
        aec_status = self.worker.get_aec_status()
        status_text = aec_status.value
        
        if aec_status.name == "SPEEX_AVAILABLE":
            color = "green"
        elif aec_status.name == "FALLBACK_ACTIVE":
            color = "orange"
        else:
            color = "gray"
        
        self.aec_status_label.config(text=status_text, foreground=color)
    
    def _start_update_loop(self):
        """Start the GUI update loop."""
        if self.worker.is_running():
            self._update_status()
        
        # Schedule next update
        self.root.after(config.GUI_UPDATE_MS, self._start_update_loop)
    
    def on_close(self):
        """Handle window close."""
        self.worker.stop()
        self.root.destroy()


def run_gui(worker: ConversationWorker, audio: AudioCapture, volume: VolumeController):
    """
    Run the GUI application.
    
    This function creates and runs the Tkinter GUI main loop, which blocks
    until the user closes the window. The worker will be automatically stopped
    when the window is closed.
    
    Args:
        worker: Conversation worker instance
        audio: Audio capture instance
        volume: Volume controller instance
        
    Returns:
        None - This function blocks until the GUI is closed
        
    Raises:
        Exception: May raise exceptions related to Tkinter initialization
                  or GUI rendering
    """
    root = tk.Tk()
    app = ConversationGUI(root, worker, audio, volume)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
