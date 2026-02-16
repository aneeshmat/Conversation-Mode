// aec_vad_v2.c
// Minimal, safe echo canceller stub compatible with aec_wrapper.py
// API:
//   void* aec_create(void);
//   void  aec_free(void* st);
//   void  aec_process_buffer(void* st,
//                            const float* ref,
//                            const float* mic,
//                            float* out,
//                            int len,
//                            int delay);

#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
  int max_len;
} AECState;

void* aec_create(void) {
  AECState* st = (AECState*)malloc(sizeof(AECState));
  if (!st) return NULL;
  st->max_len = 2048;  // safety cap; Python never passes more than 1024
  return (void*)st;
}

void aec_free(void* state) {
  if (!state) return;
  AECState* st = (AECState*)state;
  free(st);
}

void aec_process_buffer(void* state,
                        const float* ref,
                        const float* mic,
                        float* out,
                        int len,
                        int delay)
{
  if (!state || !ref || !mic || !out) {
    return;
  }
  
  AECState* st = (AECState*)state;
  
  // Clamp length to safe range
  if (len <= 0) {
    return;
  }
  if (len > st->max_len) {
    len = st->max_len;
  }
  
  // Clamp delay to a sane range
  if (delay < 0) {
    delay = 0;
  }
  if (delay > st->max_len - 1) {
    delay = st->max_len - 1;
  }
  
  // Very simple fixed echo suppression:
  // out[n] = mic[n] - k * ref[n - delay] (if in range)
  const float k = 0.5f;
  
  for (int n = 0; n < len; ++n) {
    float echo = 0.0f;
    int idx = n - delay;
    if (idx >= 0 && idx < len) {
      echo = ref[idx];
    }
    out[n] = mic[n] - k * echo;
  }
}
