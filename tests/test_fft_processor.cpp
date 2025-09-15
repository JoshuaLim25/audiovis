#include "audiovis/fft_processor.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

namespace audiovis {
namespace {

class FFTProcessorTest : public ::testing::Test {
protected:
    static constexpr std::size_t kDefaultFFTSize = 1024;
    static constexpr float kSampleRate = 48000.0f;

    /// Generates a pure sine wave at the specified frequency.
    static std::vector<float> generate_sine(
        float frequency,
        float sample_rate,
        std::size_t num_samples,
        float amplitude = 1.0f
    ) {
        std::vector<float> samples(num_samples);
        const float omega = 2.0f * std::numbers::pi_v<float> * frequency / sample_rate;

        for (std::size_t i = 0; i < num_samples; ++i) {
            samples[i] = amplitude * std::sin(omega * static_cast<float>(i));
        }

        return samples;
    }

    /// Returns the bin index with maximum magnitude.
    static std::size_t find_peak_bin(std::span<const float> magnitudes) {
        std::size_t peak_idx = 0;
        float peak_val = magnitudes[0];

        for (std::size_t i = 1; i < magnitudes.size(); ++i) {
            if (magnitudes[i] > peak_val) {
                peak_val = magnitudes[i];
                peak_idx = i;
            }
        }

        return peak_idx;
    }
};

TEST_F(FFTProcessorTest, ConstructsWithValidConfig) {
    FFTConfig config{.fft_size = 512};
    FFTProcessor proc{config};

    EXPECT_EQ(proc.fft_size(), 512);
    EXPECT_EQ(proc.bin_count(), 257);  // N/2 + 1
}

TEST_F(FFTProcessorTest, RejectNonPowerOfTwoSize) {
    FFTConfig config{.fft_size = 500};
    EXPECT_THROW(FFTProcessor{config}, std::invalid_argument);
}

TEST_F(FFTProcessorTest, BinToFrequencyCalculation) {
    FFTConfig config{.fft_size = 1024};
    FFTProcessor proc{config};

    // Bin 0 is DC (0 Hz)
    EXPECT_FLOAT_EQ(proc.bin_to_frequency(0, kSampleRate), 0.0f);

    // Bin N/2 is Nyquist (sample_rate / 2)
    EXPECT_FLOAT_EQ(proc.bin_to_frequency(512, kSampleRate), 24000.0f);

    // Frequency resolution is sample_rate / fft_size
    const float resolution = kSampleRate / 1024.0f;
    EXPECT_FLOAT_EQ(proc.bin_to_frequency(1, kSampleRate), resolution);
}

TEST_F(FFTProcessorTest, FrequencyToBinCalculation) {
    FFTConfig config{.fft_size = 1024};
    FFTProcessor proc{config};

    EXPECT_EQ(proc.frequency_to_bin(0.0f, kSampleRate), 0);
    EXPECT_EQ(proc.frequency_to_bin(24000.0f, kSampleRate), 512);

    // 1000 Hz should map to bin ~21 (1000 * 1024 / 48000 ≈ 21.3)
    EXPECT_EQ(proc.frequency_to_bin(1000.0f, kSampleRate), 21);
}

TEST_F(FFTProcessorTest, DetectsSineFrequency) {
    FFTConfig config{
        .fft_size = kDefaultFFTSize,
        .window = WindowFunction::Hann,
        .use_magnitude_db = false  // Linear for easier analysis
    };
    FFTProcessor proc{config};

    // Generate 1000 Hz sine wave
    constexpr float test_freq = 1000.0f;
    auto samples = generate_sine(test_freq, kSampleRate, kDefaultFFTSize);

    std::vector<float> magnitudes(proc.bin_count());
    proc.compute(samples, magnitudes);

    // Find the peak bin
    const std::size_t peak_bin = find_peak_bin(magnitudes);
    const float detected_freq = proc.bin_to_frequency(peak_bin, kSampleRate);

    // Should be within one bin of expected frequency
    const float resolution = kSampleRate / static_cast<float>(kDefaultFFTSize);
    EXPECT_NEAR(detected_freq, test_freq, resolution);
}

TEST_F(FFTProcessorTest, DistinguishesTwoFrequencies) {
    FFTConfig config{
        .fft_size = 2048,  // Higher resolution
        .window = WindowFunction::Hann,
        .use_magnitude_db = false
    };
    FFTProcessor proc{config};

    // Generate two sine waves
    constexpr float freq1 = 440.0f;   // A4
    constexpr float freq2 = 880.0f;   // A5 (octave above)

    auto sine1 = generate_sine(freq1, kSampleRate, 2048, 0.5f);
    auto sine2 = generate_sine(freq2, kSampleRate, 2048, 0.5f);

    // Mix them
    std::vector<float> mixed(2048);
    for (std::size_t i = 0; i < 2048; ++i) {
        mixed[i] = sine1[i] + sine2[i];
    }

    std::vector<float> magnitudes(proc.bin_count());
    proc.compute(mixed, magnitudes);

    // Find bins for both frequencies
    const std::size_t expected_bin1 = proc.frequency_to_bin(freq1, kSampleRate);
    const std::size_t expected_bin2 = proc.frequency_to_bin(freq2, kSampleRate);

    // Both should have significant energy (check they're in top 10% of max)
    float max_mag = *std::max_element(magnitudes.begin(), magnitudes.end());

    EXPECT_GT(magnitudes[expected_bin1], max_mag * 0.5f);
    EXPECT_GT(magnitudes[expected_bin2], max_mag * 0.5f);
}

TEST_F(FFTProcessorTest, DecibelConversion) {
    FFTConfig config{
        .fft_size = kDefaultFFTSize,
        .window = WindowFunction::Rectangular,
        .use_magnitude_db = true,
        .db_floor = -60.0f,
        .db_ceiling = 0.0f
    };
    FFTProcessor proc{config};

    // Full-scale sine should produce magnitude near 1.0 (normalized from 0 dB)
    auto samples = generate_sine(1000.0f, kSampleRate, kDefaultFFTSize, 1.0f);

    std::vector<float> magnitudes(proc.bin_count());
    proc.compute(samples, magnitudes);

    const std::size_t peak_bin = find_peak_bin(magnitudes);

    // Peak should be close to 1.0 (0 dB normalized to [0,1])
    EXPECT_GT(magnitudes[peak_bin], 0.8f);
}

TEST_F(FFTProcessorTest, SilenceProducesLowMagnitudes) {
    FFTConfig config{
        .fft_size = kDefaultFFTSize,
        .window = WindowFunction::Hann,
        .use_magnitude_db = true,
        .db_floor = -80.0f,
        .db_ceiling = 0.0f
    };
    FFTProcessor proc{config};

    std::vector<float> silence(kDefaultFFTSize, 0.0f);
    std::vector<float> magnitudes(proc.bin_count());

    proc.compute(silence, magnitudes);

    // All magnitudes should be at or near the floor
    for (const float mag : magnitudes) {
        EXPECT_LT(mag, 0.01f);  // Essentially zero
    }
}

TEST_F(FFTProcessorTest, WindowFunctionAffectsLeakage) {
    // Rectangular window has more spectral leakage than Hann
    auto samples = generate_sine(1000.0f, kSampleRate, kDefaultFFTSize);

    FFTConfig rect_config{
        .fft_size = kDefaultFFTSize,
        .window = WindowFunction::Rectangular,
        .use_magnitude_db = false
    };
    FFTProcessor rect_proc{rect_config};

    FFTConfig hann_config{
        .fft_size = kDefaultFFTSize,
        .window = WindowFunction::Hann,
        .use_magnitude_db = false
    };
    FFTProcessor hann_proc{hann_config};

    std::vector<float> rect_mags(rect_proc.bin_count());
    std::vector<float> hann_mags(hann_proc.bin_count());

    rect_proc.compute(samples, rect_mags);
    hann_proc.compute(samples, hann_mags);

    // Sum energy outside the main peak (leakage)
    const std::size_t peak_bin = find_peak_bin(rect_mags);
    float rect_leakage = 0.0f;
    float hann_leakage = 0.0f;

    for (std::size_t i = 0; i < rect_mags.size(); ++i) {
        // Exclude bins near the peak (±3)
        if (i < peak_bin - 3 || i > peak_bin + 3) {
            rect_leakage += rect_mags[i];
            hann_leakage += hann_mags[i];
        }
    }

    // Hann window should have less leakage
    EXPECT_LT(hann_leakage, rect_leakage);
}

TEST_F(FFTProcessorTest, ZeroPadsShortInput) {
    FFTConfig config{.fft_size = 1024};
    FFTProcessor proc{config};

    // Provide fewer samples than FFT size
    auto samples = generate_sine(1000.0f, kSampleRate, 512);

    std::vector<float> magnitudes(proc.bin_count());

    // Should not crash with short input
    EXPECT_EQ(proc.compute(samples, magnitudes), proc.bin_count());
}

TEST_F(FFTProcessorTest, LogBandMappingCoversBins) {
    constexpr std::size_t fft_size = 2048;
    constexpr std::size_t bin_count = fft_size / 2 + 1;
    constexpr std::size_t num_bars = 32;

    auto bands = compute_log_bands(
        bin_count,
        num_bars,
        20.0f,
        20000.0f,
        kSampleRate,
        fft_size
    );

    EXPECT_EQ(bands.size(), num_bars);

    // Each band should have at least one bin
    for (const auto& [lo, hi] : bands) {
        EXPECT_LT(lo, hi);
    }

    // Bands should roughly cover low to high frequency
    EXPECT_EQ(bands.front().first, 0);  // Or close to 0 for 20 Hz
}

TEST_F(FFTProcessorTest, ConfigUpdatePreservesCorrectness) {
    FFTProcessor proc{{.fft_size = 512}};

    auto samples = generate_sine(500.0f, kSampleRate, 512);
    std::vector<float> magnitudes(proc.bin_count());
    proc.compute(samples, magnitudes);

    // Update to larger FFT
    proc.set_config({.fft_size = 1024});

    samples = generate_sine(500.0f, kSampleRate, 1024);
    magnitudes.resize(proc.bin_count());
    proc.compute(samples, magnitudes);

    // Should still detect the frequency correctly
    const std::size_t peak_bin = find_peak_bin(magnitudes);
    const float detected = proc.bin_to_frequency(peak_bin, kSampleRate);
    EXPECT_NEAR(detected, 500.0f, 50.0f);  // Within reasonable tolerance
}

}  // namespace
}  // namespace audiovis
