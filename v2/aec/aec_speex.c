/*
 * SpeexDSP-based Acoustic Echo Cancellation C Wrapper
 * 
 * This is a clean C wrapper around SpeexDSP echo cancellation API.
 * 
 * Compilation:
 *   Linux:   gcc -O2 -shared -fPIC aec_speex.c -o libaec_speex.so -lspeexdsp
 *   macOS:   gcc -O2 -shared -fPIC aec_speex.c -o libaec_speex.dylib -lspeexdsp
 *   Windows: gcc -O2 -shared aec_speex.c -o libaec_speex.dll -lspeexdsp
 */

#include <stdlib.h>
#include <string.h>
#include <speex/speex_echo.h>
#include <speex/speex_preprocess.h>

/* Cross-platform export macros */
#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

typedef struct {
    SpeexEchoState *echo_state;
    SpeexPreprocessState *preprocess_state;
    int frame_size;
    int filter_length;
} AECState;

/**
 * Create and initialize AEC state.
 * 
 * @param frame_size Number of samples per frame
 * @param filter_length Length of echo cancellation filter in samples
 * @param sample_rate Sample rate in Hz
 * @return Pointer to AECState, or NULL on error
 */
EXPORT AECState* aec_create(int frame_size, int filter_length, int sample_rate) {
    AECState *state = (AECState*)malloc(sizeof(AECState));
    if (!state) {
        return NULL;
    }
    
    state->frame_size = frame_size;
    state->filter_length = filter_length;
    
    /* Initialize echo cancellation state */
    state->echo_state = speex_echo_state_init(frame_size, filter_length);
    if (!state->echo_state) {
        free(state);
        return NULL;
    }
    
    /* Set sample rate */
    speex_echo_ctl(state->echo_state, SPEEX_ECHO_SET_SAMPLING_RATE, &sample_rate);
    
    /* Initialize preprocessing (for residual echo suppression) */
    state->preprocess_state = speex_preprocess_state_init(frame_size, sample_rate);
    if (!state->preprocess_state) {
        speex_echo_state_destroy(state->echo_state);
        free(state);
        return NULL;
    }
    
    /* Link echo state to preprocessor for echo suppression */
    speex_preprocess_ctl(state->preprocess_state, SPEEX_PREPROCESS_SET_ECHO_STATE, 
                         state->echo_state);
    
    /* Enable echo suppression in preprocessor */
    int echo_suppress = 1;
    speex_preprocess_ctl(state->preprocess_state, SPEEX_PREPROCESS_SET_ECHO_SUPPRESS, 
                         &echo_suppress);
    
    /* Set aggressive echo suppression */
    int echo_suppress_active = -50;  /* dB of suppression */
    speex_preprocess_ctl(state->preprocess_state, SPEEX_PREPROCESS_SET_ECHO_SUPPRESS_ACTIVE,
                         &echo_suppress_active);
    
    return state;
}

/**
 * Process one frame of audio through AEC.
 * 
 * @param state AEC state
 * @param ref_frame Reference signal (far-end, speaker output), 16-bit samples
 * @param mic_frame Microphone signal (near-end, mic input), 16-bit samples
 * @param out_frame Output buffer for echo-cancelled signal, 16-bit samples
 * @param frame_size Number of samples in each frame (must match state->frame_size)
 */
EXPORT void aec_process(AECState *state, 
                       const short *ref_frame, 
                       const short *mic_frame,
                       short *out_frame,
                       int frame_size) {
    if (!state || frame_size != state->frame_size) {
        /* Copy mic to output if invalid state or size mismatch */
        if (mic_frame && out_frame) {
            memcpy(out_frame, mic_frame, frame_size * sizeof(short));
        }
        return;
    }
    
    /* Perform echo cancellation */
    speex_echo_cancellation(state->echo_state, mic_frame, ref_frame, out_frame);
    
    /* Apply preprocessing (residual echo suppression, etc.) */
    speex_preprocess_run(state->preprocess_state, out_frame);
}

/**
 * Reset AEC state (clears filter coefficients and history).
 * 
 * @param state AEC state
 */
EXPORT void aec_reset(AECState *state) {
    if (!state) {
        return;
    }
    
    if (state->echo_state) {
        speex_echo_state_reset(state->echo_state);
    }
}

/**
 * Destroy AEC state and free resources.
 * 
 * @param state AEC state to destroy
 */
EXPORT void aec_destroy(AECState *state) {
    if (!state) {
        return;
    }
    
    if (state->preprocess_state) {
        speex_preprocess_state_destroy(state->preprocess_state);
    }
    
    if (state->echo_state) {
        speex_echo_state_destroy(state->echo_state);
    }
    
    free(state);
}
