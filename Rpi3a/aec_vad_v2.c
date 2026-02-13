// Save as aec_core_vad.c
// Compile (Linux):
//   gcc -O3 -fPIC -shared aec_core_vad.c -o aec_vad.so -lm
// Or (Windows, DLL):
//   cl /O2 /LD aec_core_vad.c

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define FILTER_LEN 1024        // longer filter = better echo modeling
#define MAX_DELAY  8192
#define MASK       (MAX_DELAY - 1)

#define DTD_WINDOW 128         // RMS window for double-talk detection

typedef struct {
    float w[FILTER_LEN];       // adaptive filter taps
    float x[MAX_DELAY];        // circular buffer for reference
    int   pos;

    // short-term RMS history for double-talk detection
    float mic_hist[DTD_WINDOW];
    float spk_hist[DTD_WINDOW];
    int   hist_pos;
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
    const float mu   = 0.02f;    // NLMS step size
    const float leak = 0.9995f;  // leakage factor
    const float eps  = 0.001f;   // avoids divide-by-zero

    for (int j = 0; j < len; j++) {

        // Write current reference sample into circular buffer
        s->x[s->pos] = speaker_buf[j];
        int ref_idx = (s->pos - delay + MAX_DELAY) & MASK;

        // FIR echo estimate and reference energy
        float y      = 0.0f;
        float energy = eps;
        for (int i = 0; i < FILTER_LEN; i++) {
            float xi = s->x[(ref_idx - i + MAX_DELAY) & MASK];
            y      += s->w[i] * xi;
            energy += xi * xi;
        }

        // Error signal
        float e = mic_buf[j] - y;
        out_buf[j] = e;

        // --- RMS-based double-talk detection ---
        s->mic_hist[s->hist_pos] = mic_buf[j];
        s->spk_hist[s->hist_pos] = speaker_buf[j];
        s->hist_pos = (s->hist_pos + 1) % DTD_WINDOW;

        float mic_rms = 0.0f, spk_rms = 0.0f;
        for (int k = 0; k < DTD_WINDOW; k++) {
            mic_rms += s->mic_hist[k] * s->mic_hist[k];
            spk_rms += s->spk_hist[k] * s->spk_hist[k];
        }
        mic_rms = sqrtf(mic_rms / DTD_WINDOW);
        spk_rms = sqrtf(spk_rms / DTD_WINDOW);

        // Allow adaptation only when reference dominates
        int allow_adapt = (spk_rms > 0.0001f) && (mic_rms < 3.0f * spk_rms);

        if (allow_adapt) {
            float step = (mu * e) / energy;
            for (int i = 0; i < FILTER_LEN; i++) {
                float xi = s->x[(ref_idx - i + MAX_DELAY) & MASK];
                s->w[i] = (s->w[i] * leak) + (step * xi);
            }
        }

        // Advance circular buffer index
        s->pos = (s->pos + 1) & MASK;
    }
}

EXPORT void aec_free(AEC_State* s) {
    if (s) {
        free(s);
    }
}
