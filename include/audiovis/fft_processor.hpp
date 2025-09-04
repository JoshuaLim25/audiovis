#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace audiovis {

/// Window functions for spectral analysis.
/// The choice of window affects frequency resolution vs. spectral leakage tradeoff.
enum class WindowFunction {
    Rectangular,  // No windowing - maximum resolution, maximum leakage
    Hann,         // Good general purpose - moderate resolution and leakage
    Hamming,      // Similar to Hann with slightly different sidelobe behavior
    Blackman,     // Low leakage at cost of frequency resolution
    FlatTop       // Accurate amplitude measurement, poor frequency resolution
};

/// Configuration for FFT processing.
struct FFTConfig {
    std::size_t fft_size = 2048;              // Must be power of two
    WindowFunction window = WindowFunction::Hann;
    bool use_magnitude_db = true;             // Output in decibels
    float db_floor = -80.0f;                  // Minimum dB value (noise floor)
    float db_ceiling = 0.0f;                  // Maximum dB value (0 dB = full scale)
};

/// Computes FFT and extracts magnitude spectrum from audio samples.
///
/// This class manages FFTW resources and provides a simple interface for
/// real-time spectrum analysis. It maintains internal buffers for the window
/// function and FFT input/output, so repeated calls don't allocate.
///
/// Thread safety: NOT thread-safe. Create separate instances for different threads,
/// or protect access externally. Designed to be called from visualization thread only.
class FFTProcessor {
public:
    /// Constructs processor with given FFT configuration.
    /// Allocates FFTW plan and internal buffers.
    explicit FFTProcessor(const FFTConfig& config = {});

    /// Destructor releases FFTW resources.
    ~FFTProcessor();

    // Non-copyable (FFTW plans are not copyable)
    FFTProcessor(const FFTProcessor&) = delete;
    FFTProcessor& operator=(const FFTProcessor&) = delete;

    // Movable
    FFTProcessor(FFTProcessor&& other) noexcept;
    FFTProcessor& operator=(FFTProcessor&& other) noexcept;

    /// Computes magnitude spectrum from input samples.
    ///
    /// @param samples Input audio samples. If fewer than fft_size, zero-padded.
    ///                If more, only the last fft_size samples are used.
    /// @param output  Output buffer for magnitude values. Must have capacity >= bin_count().
    ///                Values are in dB if use_magnitude_db is true, otherwise linear.
    /// @return Number of magnitude values written (always bin_count() on success).
    std::size_t compute(std::span<const float> samples, std::span<float> output);

    /// Returns the number of output magnitude bins (fft_size / 2 + 1).
    [[nodiscard]] std::size_t bin_count() const noexcept { return config_.fft_size / 2 + 1; }

    /// Returns the FFT size (number of input samples processed).
    [[nodiscard]] std::size_t fft_size() const noexcept { return config_.fft_size; }

    /// Returns the frequency (Hz) corresponding to a given bin index.
    /// @param bin_index Index into magnitude output (0 to bin_count()-1)
    /// @param sample_rate The sample rate of the input audio
    [[nodiscard]] float bin_to_frequency(std::size_t bin_index, float sample_rate) const noexcept {
        return static_cast<float>(bin_index) * sample_rate / static_cast<float>(config_.fft_size);
    }

    /// Returns the bin index closest to a given frequency.
    [[nodiscard]] std::size_t frequency_to_bin(float frequency, float sample_rate) const noexcept;

    /// Provides access to configuration.
    [[nodiscard]] const FFTConfig& config() const noexcept { return config_; }

    /// Updates configuration. Reallocates buffers if fft_size changes.
    void set_config(const FFTConfig& config);

private:
    void allocate_buffers();
    void compute_window();
    void release();

    FFTConfig config_;

    // FFTW resources (opaque pointers to avoid including fftw3.h)
    struct FFTWData;
    std::unique_ptr<FFTWData> fftw_;

    // Pre-computed window coefficients
    std::vector<float> window_;
};

/// Utility: Generates logarithmically-spaced bin indices for display.
/// Useful for mapping linear FFT bins to a logarithmic frequency axis.
///
/// @param bin_count Number of FFT magnitude bins
/// @param num_bars Number of display bars/bands desired
/// @param min_freq Minimum frequency (Hz)
/// @param max_freq Maximum frequency (Hz)
/// @param sample_rate Audio sample rate (Hz)
/// @return Vector of bin index ranges, one per display bar
std::vector<std::pair<std::size_t, std::size_t>> compute_log_bands(
    std::size_t bin_count,
    std::size_t num_bars,
    float min_freq,
    float max_freq,
    float sample_rate,
    std::size_t fft_size
);

}  // namespace audiovis
