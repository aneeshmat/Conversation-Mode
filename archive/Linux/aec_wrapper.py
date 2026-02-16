import ctypes
import numpy as np

AEC_LIB_PATH = "./aec_vad.so"
AEC_DELAY = 240  # you can tune this later

class AECWrapper:
    def __init__(self, lib_path: str = AEC_LIB_PATH):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.aec_process_buffer.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.aec_create.restype = ctypes.c_void_p
        self.lib.aec_free.argtypes = [ctypes.c_void_p]
        self.state = self.lib.aec_create()

    def process(self, ref_frame: np.ndarray, mic_frame: np.ndarray) -> np.ndarray:
        # Ensure 1D float32
        mic = mic_frame.astype(np.float32).ravel()
        ref = ref_frame.astype(np.float32).ravel()

        # Match lengths (pad/truncate ref to mic length)
        n = mic.shape[0]
        if ref.shape[0] < n:
            ref_padded = np.zeros(n, dtype=np.float32)
            ref_padded[: ref.shape[0]] = ref
            ref = ref_padded
        elif ref.shape[0] > n:
            ref = ref[:n]

        out = np.zeros(n, dtype=np.float32)

        self.lib.aec_process_buffer(
            self.state,
            ref,
            mic,
            out,
            int(n),
            int(AEC_DELAY),
        )
        return out

    def close(self):
        if self.state is not None:
            self.lib.aec_free(self.state)
            self.state = None

