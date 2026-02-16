import onnxruntime as ort

print("Loading model…")
sess = ort.InferenceSession("silero_vad.onnx", providers=["CPUExecutionProvider"])

print("\n=== INPUTS ===")
for inp in sess.get_inputs():
    print("Name:", inp.name)
    print("Shape:", inp.shape)
    print("Type:", inp.type)
    print()

print("\n=== OUTPUTS ===")
for out in sess.get_outputs():
    print("Name:", out.name)
    print("Shape:", out.shape)
    print("Type:", out.type)
    print()