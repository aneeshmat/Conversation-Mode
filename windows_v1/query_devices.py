import sounddevice as sd
print(sd.query_devices())

import sounddevice as sd

def check_device_details(device_id):
    try:
        info = sd.query_devices(device_id)
        print(f"\n--- Details for Device {device_id}: {info['name']} ---")
        print(f"Max Input Channels: {info['max_input_channels']}")
        print(f"Max Output Channels: {info['max_output_channels']}")
        print(f"Default Sample Rate: {info['default_samplerate']} Hz")
        
        # Test if 2 channels at your specific rate is supported
        supported = sd.check_input_settings(device=device_id, channels=2, samplerate=info['default_samplerate'])
        print(f"Is 2-channel input supported? YES")
    except Exception as e:
        print(f"Is 2-channel input supported? NO (Error: {e})")

# Checking both your IDs from the previous list
check_device_details(30) # Your USB Mic
check_device_details(26) # Your Marshall Willen

