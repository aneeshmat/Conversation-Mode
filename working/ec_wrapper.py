import numpy as np
import ctypes
import os

class EC:
    def __init__(self, sample_rate, frame_size, filter_length=None, delay=None):
        # Path to the shared library (relative to this file)
        lib_path = os.path.join(os.path.dirname(__file__), '../ec/src/libec.so')
        self.lib = ctypes.CDLL(lib_path)
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.filter_length = filter_length if filter_length is not None else 10 * frame_size
        self.delay = delay if delay is not None else 0
        # Set argument/return types
        self.lib.ec_create.restype = ctypes.c_void_p
        self.lib.ec_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.ec_destroy.argtypes = [ctypes.c_void_p]
        self.lib.ec_process.argtypes = [ctypes.c_void_p,
                                        ctypes.POINTER(ctypes.c_short),
                                        ctypes.POINTER(ctypes.c_short),
                                        ctypes.POINTER(ctypes.c_short)]
        # Create EC instance with filter_length and delay
        self.obj = self.lib.ec_create(sample_rate, frame_size, self.filter_length, self.delay)

    def echo_cancel(self, mic, ref):
        # mic and ref must be int16 numpy arrays of length frame_size
        out = np.zeros(self.frame_size, dtype=np.int16)
        mic_ptr = mic.ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        ref_ptr = ref.ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_short))
        self.lib.ec_process(self.obj, mic_ptr, ref_ptr, out_ptr)
        return out

    def __del__(self):
        if hasattr(self, 'obj') and self.obj:
            self.lib.ec_destroy(self.obj)
