#include "aec_core.h"

#define EXPORT __declspec(dllexport)

// MAX_DELAY is large enough to handle 48kHz audio buffers
EXPORT AEC_State* aec_create() {
    AEC_State* state = (AEC_State*)malloc(sizeof(AEC_State));
    if (state) {
        memset(state->w, 0, sizeof(state->w));
        memset(state->x, 0, sizeof(state->x));
        state->pos = 0;
        state->mu = 0.005f; // Stable learning rate
    }
    return state;
}

EXPORT float aec_process(AEC_State* state, float speaker_sample, float mic_sample) {
    if (!state) return mic_sample;

    state->x[state->pos] = speaker_sample;

    // Tuning for Windows/Surface latency at 48kHz (approx 60ms)
    int delay_samples = 2880; 
    int ref_idx = (state->pos - delay_samples + MAX_DELAY) % MAX_DELAY;

    // 1. Predict Echo (FIR Filter)
    float y = 0;
    for (int i = 0; i < FILTER_LEN; i++) {
        int idx = (ref_idx - i + MAX_DELAY) % MAX_DELAY;
        y += state->w[i] * state->x[idx];
    }

    // 2. Subtraction
    float e = mic_sample - y;

    // 3. Update Weights with NLMS + Leakage
    float energy = 1e-3f; 
    for (int i = 0; i < FILTER_LEN; i++) {
        int idx = (ref_idx - i + MAX_DELAY) % MAX_DELAY;
        energy += state->x[idx] * state->x[idx];
    }

    float step = (state->mu * e) / energy;
    for (int i = 0; i < FILTER_LEN; i++) {
        int idx = (ref_idx - i + MAX_DELAY) % MAX_DELAY;
        // Leakage (0.9999) prevents the filter from exploding into feedback
        state->w[i] = (state->w[i] * 0.9999f) + (step * state->x[idx]);
    }

    state->pos = (state->pos + 1) % MAX_DELAY;
    
    // Hard clipper for safety
    if (e > 1.0f) e = 1.0f;
    if (e < -1.0f) e = -1.0f;

    return e; 
}

EXPORT void aec_free(AEC_State* state) {
    if (state) free(state);
}