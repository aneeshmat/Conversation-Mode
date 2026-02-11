#ifndef AEC_CORE_H
#define AEC_CORE_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define FILTER_LEN 1024
#define MAX_DELAY 19200

typedef struct {
    float w[FILTER_LEN];
    float x[MAX_DELAY]; 
    int pos;
    float mu;
} AEC_State;

#endif