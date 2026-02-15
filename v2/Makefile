# Makefile for building SpeexDSP AEC library
#
# Prerequisites:
#   - libspeexdsp-dev (Debian/Ubuntu: sudo apt install libspeexdsp-dev)
#   - gcc or clang
#
# Usage:
#   make            # Build the library
#   make clean      # Clean build artifacts

# Detect OS
UNAME_S := $(shell uname -s)

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -O2 -fPIC -Wall

# Linker flags
LDFLAGS = -lspeexdsp

# Source and output files
SRC = aec/aec_speex.c

ifeq ($(UNAME_S),Linux)
    TARGET = aec/libaec_speex.so
    CFLAGS += -shared
endif

ifeq ($(UNAME_S),Darwin)  # macOS
    TARGET = aec/libaec_speex.dylib
    CFLAGS += -dynamiclib
endif

ifeq ($(OS),Windows_NT)
    TARGET = aec/libaec_speex.dll
    CFLAGS += -shared
endif

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	@echo "Building AEC library for $(UNAME_S)..."
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

clean:
	rm -f $(TARGET)
	@echo "Cleaned build artifacts"

# Check if SpeexDSP is installed
check:
	@echo "Checking for SpeexDSP installation..."
	@pkg-config --exists speexdsp && echo "SpeexDSP found" || echo "SpeexDSP not found. Install with: sudo apt install libspeexdsp-dev"

.PHONY: all clean check
