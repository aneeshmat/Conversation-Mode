#include <stdlib.h>
#include <string.h>

#define FILTER_LEN 512  // Shorter is better/faster for the Pi 3A
#define MAX_DELAY 16384 
#define MASK (MAX_DELAY - 1)

typedef struct {
    float w[FILTER_LEN];
    float x[MAX_DELAY];
    int pos;
} AEC_State;

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

EXPORT AEC_State* aec_create() {
    AEC_State* state = (AEC_State*)malloc(sizeof(AEC_State));
    if (state) {
        memset(state->w, 0, sizeof(state->w));
        memset(state->x, 0, sizeof(state->x));
        state->pos = 0;
    }
    return state;
}

EXPORT void aec_process_buffer(AEC_State* state, float* speaker_buf, float* mic_buf, float* out_buf, int len, int delay) {
    float mu = 0.01f;     // Very slow, stable learning
    float leak = 0.999f; // Prevents weights from growing forever
    float eps = 0.01f; 

    for (int j = 0; j < len; j++) {
        state->x[state->pos] = speaker_buf[j];
        int ref_idx = (state->pos - delay + MAX_DELAY) & MASK;
        
        float y = 0;
        float energy = eps; 
        for (int i = 0; i < FILTER_LEN; i++) {
            float xi = state->x[(ref_idx - i + MAX_DELAY) & MASK];
            y += state->w[i] * xi;
            energy += xi * xi;
        }

        float e = mic_buf[j] - y;

        // Weight update with Leaky NLMS
        float step = (mu * e) / energy;
        for (int i = 0; i < FILTER_LEN; i++) {
            state->w[i] = (state->w[i] * leak) + (step * state->x[(ref_idx - i + MAX_DELAY) & MASK]);
        }

        out_buf[j] = e;
        state->pos = (state->pos + 1) & MASK;
    }
}

EXPORT void aec_free(AEC_State* state) { if (state) free(state); }