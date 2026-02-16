import os
import shutil
import subprocess
import re
import tkinter as tk

from audio_io import AudioIO, FRAME_SIZE, DEVICE_RATE
from worker import ConversationWorker
from gui import ConversationGUI

def _use_pactl() -> bool:
    return shutil.which("pactl") is not None

def _parse_pactl_volume(stdout: str) -> int:
    vals = [int(m.group(1)) for m in re.finditer(r'(\d+)%', stdout)]
    if vals:
        return int(round(sum(vals) / len(vals)))
    return -1

def get_volume():
    if _use_pactl():
        try:
            out = subprocess.run(
                ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                capture_output=True, text=True, timeout=1.5
            )
            if out.returncode == 0:
                v = _parse_pactl_volume(out.stdout)
                if v >= 0:
                    return v
        except Exception:
            pass
    try:
        out = subprocess.run(
            ["amixer", "get", "Master"],
            capture_output=True, text=True, timeout=1.5
        )
        if out.returncode == 0:
            m = re.search(r'\[(\d+)%\]', out.stdout)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return -1

def set_volume(percent: int) -> bool:
    if _use_pactl():
        try:
            percent = max(0, min(150, int(percent)))
            out = subprocess.run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"],
                capture_output=True, text=True, timeout=1.5
            )
            return out.returncode == 0
        except Exception:
            return False
    try:
        percent = max(0, min(100, int(percent)))
        out = subprocess.run(
            ["amixer", "sset", "Master", f"{percent}%"],
            capture_output=True, text=True, timeout=1.5
        )
        return out.returncode == 0
    except Exception:
        return False

if __name__ == "__main__":
    MIC_DEVICE_ID = int(os.getenv("MIC_DEVICE_ID", 5))
    REF_DEVICE_ID = int(os.getenv("REF_DEVICE_ID", 12))

    audio_io = AudioIO(MIC_DEVICE_ID, REF_DEVICE_ID)
    worker = ConversationWorker(audio_io, get_volume, set_volume)

    root = tk.Tk()
    app = ConversationGUI(root, worker, audio_io,
                          MIC_DEVICE_ID, REF_DEVICE_ID,
                          DEVICE_RATE, FRAME_SIZE, 16000)
    root.mainloop()