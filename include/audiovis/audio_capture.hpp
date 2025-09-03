#pragma once

#include "audiovis/ring_buffer.hpp"

#include <portaudio.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

namespace audiovis {

/// Audio capture configuration.
struct AudioConfig {
    std::uint32_t sample_rate = 48000;    // Samples per second
    std::uint32_t buffer_frames = 256;    // Frames per callback (latency tradeoff)
    std::uint32_t channels = 1;           // Mono capture for visualization
    float ring_buffer_seconds = 0.5f;     // History buffer duration
};

/// Audio capture statistics for monitoring.
struct AudioStats {
    std::uint64_t frames_captured = 0;
    std::uint64_t overruns = 0;           // Ring buffer overflows
    std::uint64_t callback_count = 0;
    float peak_amplitude = 0.0f;
};

/// Manages audio input capture via PortAudio.
///
/// Runs the PortAudio callback in a real-time thread and writes captured
/// samples into a lock-free ring buffer for consumption by the visualization
/// thread. The callback performs no allocations and no blocking operations.
class AudioCapture {
public:
    /// Initializes PortAudio and opens the default input device.
    /// @throws std::runtime_error on PortAudio initialization failure.
    explicit AudioCapture(const AudioConfig& config = {});

    /// Stops capture and releases PortAudio resources.
    ~AudioCapture();

    // Non-copyable, non-movable (owns PortAudio stream)
    AudioCapture(const AudioCapture&) = delete;
    AudioCapture& operator=(const AudioCapture&) = delete;
    AudioCapture(AudioCapture&&) = delete;
    AudioCapture& operator=(AudioCapture&&) = delete;

    /// Starts audio capture. Idempotent if already running.
    void start();

    /// Stops audio capture. Idempotent if already stopped.
    void stop();

    /// Returns true if capture is currently active.
    [[nodiscard]] bool is_running() const noexcept { return running_.load(std::memory_order_relaxed); }

    /// Returns the configured sample rate.
    [[nodiscard]] std::uint32_t sample_rate() const noexcept { return config_.sample_rate; }

    /// Returns the number of channels (typically 1 for visualization).
    [[nodiscard]] std::uint32_t channels() const noexcept { return config_.channels; }

    /// Returns current capture statistics.
    [[nodiscard]] AudioStats stats() const noexcept;

    /// Provides read access to the sample ring buffer.
    /// Consumer thread should read from this to get captured audio.
    [[nodiscard]] RingBuffer<float>& buffer() noexcept { return *ring_buffer_; }
    [[nodiscard]] const RingBuffer<float>& buffer() const noexcept { return *ring_buffer_; }

    /// Returns the name of the input device being used.
    [[nodiscard]] const std::string& device_name() const noexcept { return device_name_; }

    /// Lists available input devices. Static utility.
    [[nodiscard]] static std::vector<std::string> list_input_devices();

private:
    /// PortAudio callback - runs in real-time thread context.
    /// Must be static to match C callback signature.
    static int audio_callback(
        const void* input,
        void* output,
        unsigned long frame_count,
        const PaStreamCallbackTimeInfo* time_info,
        PaStreamCallbackFlags status_flags,
        void* user_data
    );

    /// Internal callback implementation.
    void process_audio(std::span<const float> samples, unsigned long status_flags);

    AudioConfig config_;
    std::string device_name_;
    std::unique_ptr<RingBuffer<float>> ring_buffer_;

    PaStream* stream_ = nullptr;
    std::atomic<bool> running_{false};

    // Statistics - updated atomically from callback
    std::atomic<std::uint64_t> frames_captured_{0};
    std::atomic<std::uint64_t> overruns_{0};
    std::atomic<std::uint64_t> callback_count_{0};
    std::atomic<float> peak_amplitude_{0.0f};
};

/// RAII guard for PortAudio library initialization.
/// Ensures Pa_Initialize/Pa_Terminate are called exactly once.
class PortAudioGuard {
public:
    PortAudioGuard();
    ~PortAudioGuard();

    PortAudioGuard(const PortAudioGuard&) = delete;
    PortAudioGuard& operator=(const PortAudioGuard&) = delete;

private:
    static std::atomic<int> ref_count_;
};

}  // namespace audiovis
