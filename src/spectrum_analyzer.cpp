#include "audiovis/spectrum_analyzer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace audiovis {

SpectrumAnalyzer::SpectrumAnalyzer(const AudioConfig& audio_config, const FFTConfig& fft_config,
                                   const AnalyzerConfig& analyzer_config)
    : audio_{std::make_unique<AudioCapture>(audio_config)},
      fft_{std::make_unique<FFTProcessor>(fft_config)},
      analyzer_config_{analyzer_config} {
    // Pre-allocate buffers
    sample_buffer_.resize(fft_->fft_size());
    magnitude_buffer_.resize(fft_->bin_count());
    smoothed_magnitudes_.resize(analyzer_config_.num_bands, 0.0f);
    peak_values_.resize(analyzer_config_.num_bands, 0.0f);

    recompute_band_mapping();
}

SpectrumAnalyzer::~SpectrumAnalyzer() {
    stop();
}

void SpectrumAnalyzer::start() {
    audio_->start();
}

void SpectrumAnalyzer::stop() {
    audio_->stop();
}

bool SpectrumAnalyzer::is_running() const noexcept {
    return audio_->is_running();
}

void SpectrumAnalyzer::set_config(const AnalyzerConfig& config) {
    const bool bands_changed =
        config.num_bands != analyzer_config_.num_bands ||
        config.min_frequency != analyzer_config_.min_frequency ||
        config.max_frequency != analyzer_config_.max_frequency ||
        config.logarithmic_frequency != analyzer_config_.logarithmic_frequency;

    analyzer_config_ = config;

    if (bands_changed) {
        smoothed_magnitudes_.resize(config.num_bands, 0.0f);
        peak_values_.resize(config.num_bands, 0.0f);
        recompute_band_mapping();
    }
}

void SpectrumAnalyzer::recompute_band_mapping() {
    if (analyzer_config_.logarithmic_frequency) {
        band_bins_ = compute_log_bands(
            fft_->bin_count(), analyzer_config_.num_bands, analyzer_config_.min_frequency,
            analyzer_config_.max_frequency, sample_rate(), fft_->fft_size());
    } else {
        // Linear frequency mapping
        band_bins_.clear();
        band_bins_.reserve(analyzer_config_.num_bands);

        const auto bins_per_band = fft_->bin_count() / analyzer_config_.num_bands;
        for (std::size_t i = 0; i < analyzer_config_.num_bands; ++i) {
            const auto start = i * bins_per_band;
            const auto end = (i + 1) * bins_per_band;
            band_bins_.emplace_back(start, std::min(end, fft_->bin_count()));
        }
    }
}

float SpectrumAnalyzer::compute_band_magnitude(std::size_t band_index) const {
    const auto& [bin_start, bin_end] = band_bins_[band_index];

    if (bin_start >= bin_end) {
        return 0.0f;
    }

    // Average the magnitudes in this band
    float sum = 0.0f;
    for (std::size_t i = bin_start; i < bin_end; ++i) {
        sum += magnitude_buffer_[i];
    }

    return sum / static_cast<float>(bin_end - bin_start);
}

SpectrumData SpectrumAnalyzer::update() {
    SpectrumData result;
    result.timestamp = std::chrono::steady_clock::now();
    result.magnitudes.resize(analyzer_config_.num_bands);
    result.peaks.resize(analyzer_config_.num_bands);

    // Read available samples from ring buffer
    auto& buffer = audio_->buffer();
    const auto available = buffer.size();
    const auto needed = fft_->fft_size();

    if (available < needed / 4) {
        // Not enough samples yet - return previous smoothed state
        result.magnitudes = smoothed_magnitudes_;
        result.peaks = peak_values_;
        return result;
    }

    // Read samples (taking the most recent if more than needed)
    if (available > needed) {
        buffer.discard(available - needed);
    }

    const auto read_count = buffer.peek({sample_buffer_.data(), sample_buffer_.size()});

    // Compute RMS and peak level from raw samples
    float sum_squares = 0.0f;
    float peak = 0.0f;
    for (std::size_t i = 0; i < read_count; ++i) {
        const float s = sample_buffer_[i];
        sum_squares += s * s;
        peak = std::max(peak, std::abs(s));
    }
    result.rms_level = std::sqrt(sum_squares / static_cast<float>(read_count));
    result.peak_level = peak;

    // Compute FFT
    fft_->compute({sample_buffer_.data(), read_count},
                  {magnitude_buffer_.data(), magnitude_buffer_.size()});

    // Map to display bands and apply smoothing
    for (std::size_t i = 0; i < analyzer_config_.num_bands; ++i) {
        const float raw = compute_band_magnitude(i);

        // Temporal smoothing (exponential moving average)
        const float alpha = 1.0f - analyzer_config_.smoothing_factor;
        smoothed_magnitudes_[i] =
            alpha * raw + analyzer_config_.smoothing_factor * smoothed_magnitudes_[i];

        // Peak hold with decay
        if (smoothed_magnitudes_[i] > peak_values_[i]) {
            peak_values_[i] = smoothed_magnitudes_[i];
        } else {
            peak_values_[i] *= analyzer_config_.peak_decay_rate;
        }

        result.magnitudes[i] = smoothed_magnitudes_[i];
        result.peaks[i] = peak_values_[i];
    }

    // Consume samples we've processed
    buffer.discard(read_count);

    return result;
}

}  // namespace audiovis
