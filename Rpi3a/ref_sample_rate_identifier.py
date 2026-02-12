device_id = 12  # pipewire
for sr in [48000, 44100, 32000, 22050, 16000]:
    try:
        sd.check_input_settings(device=device_id, samplerate=sr, channels=1)
        print(f"REF OK: {sr} Hz")
    except Exception as e:
        print(f"REF NO: {sr} Hz -> {e}")
