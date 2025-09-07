#pragma once

#include "audiovis/audio_capture.hpp"
#include "audiovis/fft_processor.hpp"

#include <chrono>
#include <cstddef>
#include <span>
#include <vector>

namespace audiovis {

/// Configuration for the spectrum analyzer display.
struct AnalyzerConfig {
    std::size_t num_bands = 64;           // Number of frequency bands to display
    float min_frequency = 20.0f;          // Lowest frequency (Hz)
    float max_frequency = 20000.0f;       // Highest frequency (Hz)
    float smoothing_factor = 0.7f;        // Temporal smoothing (0 = none, 1 = max)
    float peak_decay_rate = 0.95f;        // How fast peak markers fall (per frame)
    bool logarithmic_frequency = true;    // Log vs linear frequency axis
};

/// Represents the current state of spectrum analysis.
struct SpectrumData {
    std::vector<float> magnitudes;        // Current magnitude per band (0.0 to 1.0)
    std::vector<float> peaks;             // Peak hold values per band
    float rms_level = 0.0f;               // Overall RMS level (for VU meter)
    float peak_level = 0.0f;              // Recent peak level
    std::chrono::steady_clock::time_point timestamp;
};

/// High-level spectrum analyzer combining audio capture and FFT processing.
///
/// This class orchestrates the full pipeline: reading samples from the audio
/// capture ring buffer, computing FFT, mapping to display bands, and applying
/// temporal smoothing. It provides a simple interface for the visualization
/// layer to retrieve ready-to-render spectrum data.
///
/// Usage:
///   SpectrumAnalyzer analyzer;
///   analyzer.start();
///   while (running) {
///       auto data = analyzer.update();
///       render(data.magnitudes);
///   }
///   analyzer.stop();
class SpectrumAnalyzer {
public:
    /// Constructs analyzer with given configurations.
    explicit SpectrumAnalyzer(
        const AudioConfig& audio_config = {},
        const FFTConfig& fft_config = {},
        const AnalyzerConfig& analyzer_config = {}
    );

    ~SpectrumAnalyzer();

    // Non-copyable, non-movable
    SpectrumAnalyzer(const SpectrumAnalyzer&) = delete;
    SpectrumAnalyzer& operator=(const SpectrumAnalyzer&) = delete;

    /// Starts audio capture.
    void start();

    /// Stops audio capture.
    void stop();

    /// Returns true if capture is running.
    [[nodiscard]] bool is_running() const noexcept;

    /// Updates analysis and returns current spectrum data.
    /// Should be called once per frame from the visualization thread.
    /// Returns empty data if no new samples are available.
    [[nodiscard]] SpectrumData update();

    /// Provides read access to the underlying audio capture for stats.
    [[nodiscard]] const AudioCapture& audio() const noexcept { return *audio_; }

    /// Returns current analyzer configuration.
    [[nodiscard]] const AnalyzerConfig& config() const noexcept { return analyzer_config_; }

    /// Updates analyzer configuration (does not affect audio or FFT config).
    void set_config(const AnalyzerConfig& config);

    /// Returns the sample rate being used.
    [[nodiscard]] float sample_rate() const noexcept {
        return static_cast<float>(audio_->sample_rate());
    }

private:
    void recompute_band_mapping();
    float compute_band_magnitude(std::size_t band_index) const;

    std::unique_ptr<AudioCapture> audio_;
    std::unique_ptr<FFTProcessor> fft_;
    AnalyzerConfig analyzer_config_;

    // Internal buffers (pre-allocated to avoid runtime allocation)
    std::vector<float> sample_buffer_;      // Samples for FFT input
    std::vector<float> magnitude_buffer_;   // Raw FFT output
    std::vector<float> smoothed_magnitudes_; // Temporally smoothed values
    std::vector<float> peak_values_;        // Peak hold per band

    // Band mapping: which FFT bins contribute to each display band
    std::vector<std::pair<std::size_t, std::size_t>> band_bins_;
};

}  // namespace audiovis
