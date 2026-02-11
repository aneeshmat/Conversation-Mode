#include <stdlib.h>
#include <string.h>

#define FILTER_LEN 1024 // Double the memory
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
    float mu = 0.5f; // Aggressive learning

    for (int j = 0; j < len; j++) {
        state->x[state->pos] = speaker_buf[j];
        
        float y = 0;
        int ref_idx = (state->pos - delay + MAX_DELAY) & MASK;
        
        // Predict the echo
        for (int i = 0; i < FILTER_LEN; i++) {
            y += state->w[i] * state->x[(ref_idx - i + MAX_DELAY) & MASK];
        }

        float e = mic_buf[j] - y;
        out_buf[j] = e;

        // Update weights aggressively
        // We add a tiny limit to keep it from exploding
        for (int i = 0; i < FILTER_LEN; i++) {
            state->w[i] += mu * e * state->x[(ref_idx - i + MAX_DELAY) & MASK];
            if (state->w[i] > 5.0f) state->w[i] = 5.0f;
            if (state->w[i] < -5.0f) state->w[i] = -5.0f;
        }

        state->pos = (state->pos + 1) & MASK;
    }
}

EXPORT void aec_free(AEC_State* state) { if (state) free(state); }