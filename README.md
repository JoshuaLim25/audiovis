# audiovis

Real-time audio spectrum visualizer for Linux. Captures microphone input and renders a frequency spectrum display directly in your terminal.

![Build Status](https://github.com/yourusername/audiovis/actions/workflows/ci.yml/badge.svg)

## Quick Start

```bash
# Install dependencies (Arch Linux)
sudo pacman -S cmake portaudio fftw ncurses

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Run
./build/audiovis
```

Press `q` or `Escape` to quit.

## Architecture

The system consists of four decoupled layers communicating through lock-free data structures:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  AudioCapture   │────▶│  RingBuffer     │────▶│  FFTProcessor   │
│  (PortAudio)    │     │  (Lock-free)    │     │  (FFTW3)        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
       │                                               │
       │ Real-time thread                              │ Visualization thread
       │                                               ▼
       │                                        ┌─────────────────┐
       │                                        │ SpectrumAnalyzer│
       │                                        │ (Band mapping)  │
       │                                        └────────┬────────┘
       │                                                 │
       └─────────────────────────────────────────────────┼────────────────────┐
                                                         ▼                    │
                                                  ┌─────────────────┐         │
                                                  │ TerminalRenderer│         │
                                                  │ (ncurses)       │◀────────┘
                                                  └─────────────────┘    Stats
```

**AudioCapture** runs PortAudio's callback in a **real-time thread context**, meaning the callback must never block or allocate memory. Samples flow through a **single-producer single-consumer (SPSC) ring buffer** using acquire-release semantics to synchronize without locks.

**FFTProcessor** wraps FFTW3 with pre-allocated buffers and configurable window functions. The **Hann window** provides a reasonable tradeoff between frequency resolution and spectral leakage for music and environmental sound.

**SpectrumAnalyzer** maps linear FFT bins to logarithmically-spaced display bands and applies temporal smoothing via exponential moving average.

## Configuration

Default parameters are tuned for general use but can be adjusted in `src/terminal_renderer.cpp`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 48000 Hz | Audio capture sample rate |
| `fft_size` | 2048 | FFT window size (frequency resolution) |
| `num_bands` | 64 | Display frequency bands |
| `smoothing_factor` | 0.6 | Temporal smoothing (0=none, 1=max) |
| `db_floor` | -60 dB | Noise floor threshold |

Larger `fft_size` improves frequency resolution but increases latency. The **frequency resolution** is `sample_rate / fft_size`—with defaults, that's approximately 23 Hz per bin.

## Dependencies

| Library | Purpose | Package (Arch) |
|---------|---------|----------------|
| PortAudio | Cross-platform audio capture | `portaudio` |
| FFTW3 | Fast Fourier Transform | `fftw` |
| ncurses | Terminal rendering | `ncurses` |
| CMake ≥3.20 | Build system | `cmake` |

## Build Options

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \      # Release|Debug|RelWithDebInfo
  -DAUDIOVIS_BUILD_TESTS=ON \       # Build unit tests
  -DAUDIOVIS_ENABLE_SANITIZERS=ON \ # ASan/UBSan in Debug builds
  -DAUDIOVIS_USE_TERMINAL=ON        # Terminal UI (vs SDL2)
```

## Testing

```bash
cmake --build build --target test
# or
cd build && ctest --output-on-failure
```

Tests cover the lock-free ring buffer (including concurrent stress tests) and FFT correctness (sine wave detection, window function behavior).

## Project Structure

```
audiovis/
├── include/audiovis/
│   ├── ring_buffer.hpp       # Lock-free SPSC queue
│   ├── audio_capture.hpp     # PortAudio wrapper
│   ├── fft_processor.hpp     # FFTW3 wrapper
│   └── spectrum_analyzer.hpp # High-level coordinator
├── src/
│   ├── audio_capture.cpp
│   ├── fft_processor.cpp
│   ├── spectrum_analyzer.cpp
│   └── terminal_renderer.cpp # ncurses visualization + main()
├── tests/
│   ├── test_ring_buffer.cpp
│   └── test_fft_processor.cpp
├── .github/workflows/
│   └── ci.yml                # GitHub Actions CI
└── CMakeLists.txt
```

