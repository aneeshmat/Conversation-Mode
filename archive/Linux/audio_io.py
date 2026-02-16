import numpy as np
import sounddevice as sd
import queue
import time

FRAME_SIZE = 1024
DEVICE_RATE = 48000


class AudioIO:
    def __init__(self, mic_id: int, ref_id: int):
        self.mic_id = mic_id
        self.ref_id = ref_id

        # Reference buffer always kept at FRAME_SIZE
        self.ref_buffer = np.zeros(FRAME_SIZE, dtype=np.float32)

        # Queue for worker thread
        self.q = queue.Queue(maxsize=16)

        self.mic_stream = None
        self.ref_stream = None

    # -----------------------------
    # Reference (speaker) callback
    # -----------------------------
    def _ref_callback(self, indata, frames, time_info, status):
        if status:
            print("Ref status:", status)

        ref = indata[:, 0].astype(np.float32)

        # Normalize to FRAME_SIZE
        if ref.size >= FRAME_SIZE:
            self.ref_buffer[:] = ref[:FRAME_SIZE]
        else:
            self.ref_buffer[:ref.size] = ref
            self.ref_buffer[ref.size:] = 0.0

    # -----------------------------
    # Microphone callback
    # -----------------------------
    def _mic_callback(self, indata, frames, time_info, status):
        if status:
            print("Mic status:", status)

        mic = indata[:, 0].astype(np.float32)

        # Normalize mic to FRAME_SIZE
        if mic.size >= FRAME_SIZE:
            mic_frame = mic[:FRAME_SIZE].copy()
        else:
            mic_frame = np.zeros(FRAME_SIZE, dtype=np.float32)
            mic_frame[:mic.size] = mic

        ref_frame = self.ref_buffer.copy()

        # Push to queue
        try:
            self.q.put_nowait((mic_frame, ref_frame))
        except queue.Full:
            try:
                _ = self.q.get_nowait()
                self.q.put_nowait((mic_frame, ref_frame))
            except Exception:
                pass

    # -----------------------------
    # Start both streams
    # -----------------------------
    def start(self):
        self.mic_stream = sd.InputStream(
            device=self.mic_id,
            channels=1,
            samplerate=DEVICE_RATE,
            callback=self._mic_callback,
            blocksize=FRAME_SIZE,   # may be ignored, but we normalize anyway
            dtype='float32',
            latency="high",
        )

        self.ref_stream = sd.InputStream(
            device=self.ref_id,
            channels=1,
            samplerate=DEVICE_RATE,
            callback=self._ref_callback,
            blocksize=FRAME_SIZE,
            dtype='float32',
            latency="high",
        )

        self.mic_stream.start()
        self.ref_stream.start()

    # -----------------------------
    # Stop streams
    # -----------------------------
    def stop(self):
        try:
            if self.mic_stream:
                self.mic_stream.stop()
                self.mic_stream.close()
            if self.ref_stream:
                self.ref_stream.stop()
                self.ref_stream.close()
        except Exception:
            pass

        self.mic_stream = None
        self.ref_stream = None

        # Allow ALSA/PulseAudio to release the device before restart
        time.sleep(0.3)

    # -----------------------------
    # Queue accessor
    # -----------------------------
    def get_queue(self):
        return self.q

