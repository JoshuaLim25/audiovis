#include "audiovis/audio_capture.hpp"

#include <portaudio.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace audiovis {

// Static reference count for PortAudio library initialization
std::atomic<int> PortAudioGuard::ref_count_{0};

PortAudioGuard::PortAudioGuard() {
    if (ref_count_.fetch_add(1, std::memory_order_acq_rel) == 0) {
        PaError err = Pa_Initialize();
        if (err != paNoError) {
            ref_count_.fetch_sub(1, std::memory_order_acq_rel);
            throw std::runtime_error(std::string("Failed to initialize PortAudio: ") +
                                     Pa_GetErrorText(err));
        }
    }
}

PortAudioGuard::~PortAudioGuard() {
    if (ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        Pa_Terminate();
    }
}

// Global guard ensures PortAudio stays initialized for duration of program
static PortAudioGuard g_portaudio_guard;

AudioCapture::AudioCapture(const AudioConfig& config) : config_{config} {
    // Calculate ring buffer size from duration
    const auto buffer_samples = static_cast<std::size_t>(config_.ring_buffer_seconds *
                                                         static_cast<float>(config_.sample_rate) *
                                                         static_cast<float>(config_.channels));
    ring_buffer_ = std::make_unique<RingBuffer<float>>(buffer_samples);

    // Get default input device info
    PaDeviceIndex device = Pa_GetDefaultInputDevice();
    if (device == paNoDevice) {
        throw std::runtime_error("No default audio input device available");
    }

    const PaDeviceInfo* device_info = Pa_GetDeviceInfo(device);
    if (device_info == nullptr) {
        throw std::runtime_error("Failed to get input device info");
    }
    device_name_ = device_info->name;

    // Configure input stream parameters
    PaStreamParameters input_params{};
    input_params.device = device;
    input_params.channelCount = static_cast<int>(config_.channels);
    input_params.sampleFormat = paFloat32;
    input_params.suggestedLatency = device_info->defaultLowInputLatency;
    input_params.hostApiSpecificStreamInfo = nullptr;

    // Open the stream
    PaError err = Pa_OpenStream(&stream_, &input_params,
                                nullptr,  // No output
                                static_cast<double>(config_.sample_rate), config_.buffer_frames,
                                paClipOff,  // Don't clip samples
                                &AudioCapture::audio_callback, this);

    if (err != paNoError) {
        throw std::runtime_error(std::string("Failed to open audio stream: ") +
                                 Pa_GetErrorText(err));
    }
}

AudioCapture::~AudioCapture() {
    stop();
    if (stream_ != nullptr) {
        Pa_CloseStream(stream_);
    }
}

void AudioCapture::start() {
    if (running_.load(std::memory_order_relaxed)) {
        return;  // Already running
    }

    PaError err = Pa_StartStream(stream_);
    if (err != paNoError) {
        throw std::runtime_error(std::string("Failed to start audio stream: ") +
                                 Pa_GetErrorText(err));
    }

    running_.store(true, std::memory_order_release);
}

void AudioCapture::stop() {
    if (!running_.load(std::memory_order_relaxed)) {
        return;  // Already stopped
    }

    running_.store(false, std::memory_order_release);
    Pa_StopStream(stream_);
}

AudioStats AudioCapture::stats() const noexcept {
    return AudioStats{.frames_captured = frames_captured_.load(std::memory_order_relaxed),
                      .overruns = overruns_.load(std::memory_order_relaxed),
                      .callback_count = callback_count_.load(std::memory_order_relaxed),
                      .peak_amplitude = peak_amplitude_.load(std::memory_order_relaxed)};
}

int AudioCapture::audio_callback(const void* input, void* /*output*/, unsigned long frame_count,
                                 const PaStreamCallbackTimeInfo* /*time_info*/,
                                 PaStreamCallbackFlags status_flags, void* user_data) {
    auto* self = static_cast<AudioCapture*>(user_data);

    // Cast input buffer to float samples
    const auto* samples = static_cast<const float*>(input);
    const auto sample_count = frame_count * self->config_.channels;

    self->process_audio(std::span<const float>{samples, sample_count}, status_flags);

    return paContinue;
}

void AudioCapture::process_audio(std::span<const float> samples, unsigned long status_flags) {
    // Check for input overflow (samples were dropped before reaching us)
    if ((status_flags & paInputOverflow) != 0) {
        overruns_.fetch_add(1, std::memory_order_relaxed);
    }

    // Track peak amplitude for level metering
    float peak = 0.0f;
    for (const float sample : samples) {
        peak = std::max(peak, std::abs(sample));
    }

    // Update peak using atomic max operation
    float current_peak = peak_amplitude_.load(std::memory_order_relaxed);
    while (peak > current_peak &&
           !peak_amplitude_.compare_exchange_weak(current_peak, peak, std::memory_order_relaxed)) {
        // Retry if another update happened
    }

    // Write samples to ring buffer
    std::size_t written = ring_buffer_->try_push(samples);
    if (written < samples.size()) {
        // Ring buffer full - count as overrun
        overruns_.fetch_add(1, std::memory_order_relaxed);
    }

    // Update statistics
    frames_captured_.fetch_add(samples.size(), std::memory_order_relaxed);
    callback_count_.fetch_add(1, std::memory_order_relaxed);
}

std::vector<std::string> AudioCapture::list_input_devices() {
    std::vector<std::string> devices;

    int device_count = Pa_GetDeviceCount();
    if (device_count < 0) {
        throw std::runtime_error("Failed to enumerate audio devices");
    }

    for (int i = 0; i < device_count; ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (info != nullptr && info->maxInputChannels > 0) {
            devices.emplace_back(info->name);
        }
    }

    return devices;
}

}  // namespace audiovis
