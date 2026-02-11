#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include <stdio.h>

void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    // This is the "Wire". We copy mic input to speaker output.
    // In a real AEC, this is where the subtraction math happens.
    float* out = (float*)pOutput;
    float* in  = (float*)pInput;

    for (ma_uint32 i = 0; i < frameCount * pDevice->playback.channels; ++i) {
        out[i] = in[i]; 
    }
}

int main() {
    ma_device_config config = ma_device_config_init(ma_device_type_duplex);
    config.playback.format   = ma_format_f32;
    config.capture.format    = ma_format_f32;
    config.sampleRate        = 44100;
    config.dataCallback      = data_callback;

    ma_device device;
    if (ma_device_init(NULL, &config, &device) != MA_SUCCESS) return -1;
    ma_device_start(&device);

    printf("Loopback active. Press Enter to quit...\n");
    getchar();

    ma_device_uninit(&device);
    return 0;
}