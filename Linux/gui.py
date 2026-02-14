import tkinter as tk
from tkinter import ttk

class ConversationGUI:
    def __init__(self, root, worker, audio_io, mic_id, ref_id,
                 device_rate, frame_size, sample_rate):
        self.root = root
        self.worker = worker
        self.audio_io = audio_io
        self.running = False

        root.title("AEC + VAD + ML Classifier (Linux)")
        root.geometry("580x260")

        self.btn = ttk.Button(root, text="Start Conversation Mode",
                              command=self.toggle)
        self.btn.pack(pady=16)

        self.prob_lbl = ttk.Label(root, text="Speech Prob (smoothed): 0%")
        self.prob_lbl.pack()

        status = (
            f"MIC={mic_id}  REF={ref_id}  "
            f"I/O={device_rate}Hz  VAD={sample_rate}Hz  "
            f"Block={frame_size}"
        )
        self.status_lbl = ttk.Label(root, text=status)
        self.status_lbl.pack(pady=8)

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_ui()

    def toggle(self):
        if not self.running:
            self.audio_io.start()
            self.worker.start()
            self.btn.config(text="Stop")
            self.running = True
        else:
            self.worker.stop()
            self.audio_io.stop()
            self.btn.config(text="Start Conversation Mode")
            self.running = False

    def update_ui(self):
        pct = int(max(0.0, min(1.0, self.worker.get_vad_prob())) * 100)
        self.prob_lbl.config(text=f"Speech Prob (smoothed): {pct}%")
        self.root.after(100, self.update_ui)

    def on_close(self):
        if self.running:
            self.worker.stop()
            self.audio_io.stop()
        self.root.destroy()
