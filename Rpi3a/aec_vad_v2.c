// Save as aec_core_vad.c and compile to aec_vad.so (Linux) or aec_core_vad.dll (Windows)
#include <stdlib.h>
#include <string.h>

#define FILTER_LEN 512
#define MAX_DELAY  8192
#define MASK       (MAX_DELAY - 1)

typedef struct {
    float w[FILTER_LEN];
    float x[MAX_DELAY];
    int   pos;
} AEC_State;

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

EXPORT AEC_State* aec_create() {
    AEC_State* state = (AEC_State*)malloc(sizeof(AEC_State));
    if (state) {
        memset(state, 0, sizeof(AEC_State));
    }
    return state;
}

EXPORT void aec_process_buffer(AEC_State* state,
                               float* speaker_buf,
                               float* mic_buf,
                               float* out_buf,
                               int len,
                               int delay) {
    float mu   = 0.02f;
    float leak = 0.9995f;
    float eps  = 0.001f;

    for (int j = 0; j < len; j++) {
        // write current speaker sample into circular buffer
        state->x[state->pos] = speaker_buf[j];

        // delayed reference index
        int ref_idx = (state->pos - delay + MAX_DELAY) & MASK;

        // compute filter output (predicted echo) and reference energy
        float y      = 0.0f;
        float energy = eps;
        for (int i = 0; i < FILTER_LEN; i++) {
            float xi = state->x[(ref_idx - i + MAX_DELAY) & MASK];
            y      += state->w[i] * xi;
            energy += xi * xi;
        }

        // error = mic - predicted echo
        float e = mic_buf[j] - y;
        out_buf[j] = e;

        // --- crude double-talk protection ---
        // instantaneous powers
        float spk     = speaker_buf[j];
        float mic     = mic_buf[j];
        float spk_pow = spk * spk;
        float mic_pow = mic * mic;

        // if mic power is much larger than speaker power,
        // assume near-end speech → skip adaptation
        int allow_adapt = (spk_pow > 0.0f) && (mic_pow < 4.0f * spk_pow);

        if (allow_adapt) {
            float step = (mu * e) / energy;
            for (int i = 0; i < FILTER_LEN; i++) {
                float xi = state->x[(ref_idx - i + MAX_DELAY) & MASK];
                state->w[i] = (state->w[i] * leak) + (step * xi);
            }
        }

        // advance circular buffer position
        state->pos = (state->pos + 1) & MASK;
    }
}

EXPORT void aec_free(AEC_State* state) {
    if (state) {
        free(state);
    }
}
