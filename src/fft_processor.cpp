#include "audiovis/fft_processor.hpp"

#include <fftw3.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace audiovis {

/// Internal FFTW data structures (hidden from header).
struct FFTProcessor::FFTWData {
    fftwf_plan plan = nullptr;
    float* input = nullptr;           // FFTW-aligned input buffer
    fftwf_complex* output = nullptr;  // FFTW-aligned output buffer

    ~FFTWData() {
        if (plan != nullptr) {
            fftwf_destroy_plan(plan);
        }
        if (input != nullptr) {
            fftwf_free(input);
        }
        if (output != nullptr) {
            fftwf_free(output);
        }
    }
};

FFTProcessor::FFTProcessor(const FFTConfig& config)
    : config_{config}, fftw_{std::make_unique<FFTWData>()} {
    if ((config_.fft_size & (config_.fft_size - 1)) != 0) {
        throw std::invalid_argument("FFT size must be a power of two");
    }

    allocate_buffers();
    compute_window();
}

FFTProcessor::~FFTProcessor() = default;

FFTProcessor::FFTProcessor(FFTProcessor&& other) noexcept
    : config_{other.config_}, fftw_{std::move(other.fftw_)}, window_{std::move(other.window_)} {}

FFTProcessor& FFTProcessor::operator=(FFTProcessor&& other) noexcept {
    if (this != &other) {
        config_ = other.config_;
        fftw_ = std::move(other.fftw_);
        window_ = std::move(other.window_);
    }
    return *this;
}

void FFTProcessor::allocate_buffers() {
    release();

    const auto n = static_cast<int>(config_.fft_size);
    const auto out_size = config_.fft_size / 2 + 1;

    fftw_ = std::make_unique<FFTWData>();

    // Allocate FFTW-aligned buffers for SIMD optimization
    fftw_->input = fftwf_alloc_real(static_cast<size_t>(n));
    fftw_->output = fftwf_alloc_complex(out_size);

    if (fftw_->input == nullptr || fftw_->output == nullptr) {
        throw std::runtime_error("Failed to allocate FFTW buffers");
    }

    // Create plan (FFTW_MEASURE is slower to plan but faster to execute)
    // Use FFTW_ESTIMATE for faster startup at cost of slightly slower FFT
    fftw_->plan = fftwf_plan_dft_r2c_1d(n, fftw_->input, fftw_->output, FFTW_ESTIMATE);

    if (fftw_->plan == nullptr) {
        throw std::runtime_error("Failed to create FFTW plan");
    }

    window_.resize(config_.fft_size);
}

void FFTProcessor::compute_window() {
    const auto n = config_.fft_size;
    constexpr auto pi = std::numbers::pi_v<float>;

    switch (config_.window) {
        case WindowFunction::Rectangular:
            std::fill(window_.begin(), window_.end(), 1.0f);
            break;

        case WindowFunction::Hann:
            for (std::size_t i = 0; i < n; ++i) {
                const auto x = static_cast<float>(i) / static_cast<float>(n - 1);
                window_[i] = 0.5f * (1.0f - std::cos(2.0f * pi * x));
            }
            break;

        case WindowFunction::Hamming:
            for (std::size_t i = 0; i < n; ++i) {
                const auto x = static_cast<float>(i) / static_cast<float>(n - 1);
                window_[i] = 0.54f - 0.46f * std::cos(2.0f * pi * x);
            }
            break;

        case WindowFunction::Blackman:
            for (std::size_t i = 0; i < n; ++i) {
                const auto x = static_cast<float>(i) / static_cast<float>(n - 1);
                window_[i] =
                    0.42f - 0.5f * std::cos(2.0f * pi * x) + 0.08f * std::cos(4.0f * pi * x);
            }
            break;

        case WindowFunction::FlatTop:
            for (std::size_t i = 0; i < n; ++i) {
                const auto x = static_cast<float>(i) / static_cast<float>(n - 1);
                window_[i] = 0.21557895f - 0.41663158f * std::cos(2.0f * pi * x) +
                             0.277263158f * std::cos(4.0f * pi * x) -
                             0.083578947f * std::cos(6.0f * pi * x) +
                             0.006947368f * std::cos(8.0f * pi * x);
            }
            break;
    }
}

void FFTProcessor::release() {
    fftw_.reset();
}

std::size_t FFTProcessor::compute(std::span<const float> samples, std::span<float> output) {
    assert(fftw_ && fftw_->plan);
    assert(output.size() >= bin_count());

    const auto n = config_.fft_size;

    // Copy samples to input buffer with windowing
    // Zero-pad if fewer samples than FFT size
    const auto copy_count = std::min(samples.size(), n);
    const auto offset = n - copy_count;  // Right-align samples

    // Zero the beginning if zero-padding needed
    for (std::size_t i = 0; i < offset; ++i) {
        fftw_->input[i] = 0.0f;
    }

    // Copy and apply window
    for (std::size_t i = 0; i < copy_count; ++i) {
        fftw_->input[offset + i] = samples[samples.size() - copy_count + i] * window_[offset + i];
    }

    // Execute FFT
    fftwf_execute(fftw_->plan);

    // Compute magnitudes
    const auto num_bins = bin_count();
    const auto scale = 2.0f / static_cast<float>(n);  // Normalize

    for (std::size_t i = 0; i < num_bins; ++i) {
        const float re = fftw_->output[i][0];
        const float im = fftw_->output[i][1];
        float magnitude = std::sqrt(re * re + im * im) * scale;

        // DC and Nyquist bins don't need 2x scaling
        if (i == 0 || i == num_bins - 1) {
            magnitude *= 0.5f;
        }

        if (config_.use_magnitude_db) {
            // Convert to decibels
            constexpr float epsilon = 1e-10f;  // Avoid log(0)
            float db = 20.0f * std::log10(magnitude + epsilon);
            db = std::clamp(db, config_.db_floor, config_.db_ceiling);

            // Normalize to 0.0 - 1.0 range
            output[i] = (db - config_.db_floor) / (config_.db_ceiling - config_.db_floor);
        } else {
            output[i] = magnitude;
        }
    }

    return num_bins;
}

std::size_t FFTProcessor::frequency_to_bin(float frequency, float sample_rate) const noexcept {
    const auto bin = static_cast<std::size_t>(
        frequency * static_cast<float>(config_.fft_size) / sample_rate + 0.5f);
    return std::min(bin, bin_count() - 1);
}

void FFTProcessor::set_config(const FFTConfig& config) {
    const bool size_changed = config.fft_size != config_.fft_size;
    config_ = config;

    if (size_changed) {
        allocate_buffers();
    }
    compute_window();
}

std::vector<std::pair<std::size_t, std::size_t>> compute_log_bands(std::size_t bin_count,
                                                                   std::size_t num_bars,
                                                                   float min_freq, float max_freq,
                                                                   float sample_rate,
                                                                   std::size_t fft_size) {
    std::vector<std::pair<std::size_t, std::size_t>> bands;
    bands.reserve(num_bars);

    const float log_min = std::log10(min_freq);
    const float log_max = std::log10(max_freq);
    const float log_step = (log_max - log_min) / static_cast<float>(num_bars);

    auto freq_to_bin = [&](float freq) -> std::size_t {
        const auto bin =
            static_cast<std::size_t>(freq * static_cast<float>(fft_size) / sample_rate);
        return std::clamp(bin, std::size_t{0}, bin_count - 1);
    };

    for (std::size_t i = 0; i < num_bars; ++i) {
        const float freq_lo = std::pow(10.0f, log_min + log_step * static_cast<float>(i));
        const float freq_hi = std::pow(10.0f, log_min + log_step * static_cast<float>(i + 1));

        std::size_t bin_lo = freq_to_bin(freq_lo);
        std::size_t bin_hi = freq_to_bin(freq_hi);

        // Ensure at least one bin per band
        if (bin_hi <= bin_lo) {
            bin_hi = bin_lo + 1;
        }
        bin_hi = std::min(bin_hi, bin_count);

        bands.emplace_back(bin_lo, bin_hi);
    }

    return bands;
}

}  // namespace audiovis
