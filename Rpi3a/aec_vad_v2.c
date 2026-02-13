// Save as aec_core_vad.c and compile with:
// gcc -O3 -fPIC -shared aec_core_vad.c -o aec_vad.so -lm

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define FILTER_LEN 1024
#define MAX_DELAY  8192
#define MASK       (MAX_DELAY - 1)
#define DTD_WINDOW 128

typedef struct {
    float w[FILTER_LEN];
    float x[MAX_DELAY];
    int pos;

    float mic_hist[DTD_WINDOW];
    float spk_hist[DTD_WINDOW];
    int hist_pos;
} AEC_State;

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT AEC_State* aec_create() {
    AEC_State* s = (AEC_State*)malloc(sizeof(AEC_State));
    if (s) {
        memset(s, 0, sizeof(AEC_State));
    }
    return s;
}

EXPORT void aec_process_buffer(AEC_State* s,
                               float* speaker_buf,
                               float* mic_buf,
                               float* out_buf,
                               int len,
                               int delay) {
    float mu   = 0.02f;
    float leak = 0.9995f;
    float eps  = 0.001f;

    for (int j = 0; j < len; j++) {

        // Write reference sample
        s->x[s->pos] = speaker_buf[j];
        int ref_idx = (s->pos - delay + MAX_DELAY) & MASK;

        // FIR echo estimate
        float y = 0.0f;
        float energy = eps;
        for (int i = 0; i < FILTER_LEN; i++) {
            float xi = s->x[(ref_idx - i + MAX_DELAY) & MASK];
            y += s->w[i] * xi;
            energy += xi * xi;
        }

        float e = mic_buf[j] - y;
        out_buf[j] = e;

        // Update RMS history
        s->mic_hist[s->hist_pos] = mic_buf[j];
        s->spk_hist[s->hist_pos] = speaker_buf[j];
        s->hist_pos = (s->hist_pos + 1) % DTD_WINDOW;

        // Compute RMS
        float mic_rms = 0.0f, spk_rms = 0.0f;
        for (int k = 0; k < DTD_WINDOW; k++) {
            mic_rms += s->mic_hist[k] * s->mic_hist[k];
            spk_rms += s->spk_hist[k] * s->spk_hist[k];
        }
        mic_rms = sqrtf(mic_rms / DTD_WINDOW);
        spk_rms = sqrtf(spk_rms / DTD_WINDOW);

        // Double-talk protection
        int allow_adapt = (spk_rms > 0.0001f) && (mic_rms < 3.0f * spk_rms);

        if (allow_adapt) {
            float step = (mu * e) / energy;
            for (int i = 0; i < FILTER_LEN; i++) {
                float xi = s->x[(ref_idx - i + MAX_DELAY) & MASK];
                s->w[i] = (s->w[i] * leak) + (step * xi);
            }
        }

        s->pos = (s->pos + 1) & MASK;
    }
}

EXPORT void aec_free(AEC_State* s) {
    if (s) free(s);
}
