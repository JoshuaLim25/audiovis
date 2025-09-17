#include "audiovis/spectrum_analyzer.hpp"

#include <ncurses.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <thread>

namespace audiovis {

/// Terminal-based spectrum visualizer using ncurses.
///
/// Renders a real-time bar graph where each bar represents a frequency band.
/// Supports color gradients, peak indicators, and adaptive sizing to terminal
/// dimensions. The render loop targets 60 FPS but gracefully degrades on
/// slower terminals.
class TerminalRenderer {
public:
    TerminalRenderer() : running_{true} { init_ncurses(); }

    ~TerminalRenderer() { shutdown_ncurses(); }

    /// Main render loop. Blocks until user quits (q or Ctrl+C).
    void run(SpectrumAnalyzer& analyzer) {
        analyzer.start();

        constexpr auto frame_duration = std::chrono::milliseconds(16);  // ~60 FPS

        while (running_) {
            auto frame_start = std::chrono::steady_clock::now();

            // Handle input
            int ch = getch();
            if (ch == 'q' || ch == 'Q' || ch == 27) {  // q or Escape
                break;
            }

            // Handle terminal resize
            if (ch == KEY_RESIZE) {
                handle_resize();
            }

            // Update spectrum data
            auto data = analyzer.update();

            // Render frame
            render(data, analyzer.audio().stats());

            // Frame rate limiting
            auto frame_end = std::chrono::steady_clock::now();
            auto elapsed = frame_end - frame_start;
            if (elapsed < frame_duration) {
                std::this_thread::sleep_for(frame_duration - elapsed);
            }
        }

        analyzer.stop();
    }

    void stop() { running_ = false; }

private:
    void init_ncurses() {
        initscr();
        cbreak();
        noecho();
        curs_set(0);            // Hide cursor
        nodelay(stdscr, TRUE);  // Non-blocking getch
        keypad(stdscr, TRUE);   // Enable special keys

        // Initialize colors if supported
        if (has_colors()) {
            start_color();
            use_default_colors();

            // Color pairs for magnitude gradient (low to high)
            init_pair(1, COLOR_BLUE, -1);
            init_pair(2, COLOR_CYAN, -1);
            init_pair(3, COLOR_GREEN, -1);
            init_pair(4, COLOR_YELLOW, -1);
            init_pair(5, COLOR_RED, -1);
            init_pair(6, COLOR_MAGENTA, -1);  // Peak indicator

            has_color_ = true;
        }

        getmaxyx(stdscr, term_height_, term_width_);
    }

    void shutdown_ncurses() { endwin(); }

    void handle_resize() {
        endwin();
        refresh();
        getmaxyx(stdscr, term_height_, term_width_);
    }

    void render(const SpectrumData& data, const AudioStats& stats) {
        erase();

        // Reserve space for header and footer
        constexpr int header_lines = 2;
        constexpr int footer_lines = 2;
        const int viz_height = term_height_ - header_lines - footer_lines;
        const int viz_width = term_width_ - 2;  // 1 char margin each side

        if (viz_height < 3 || viz_width < 10) {
            mvprintw(0, 0, "Terminal too small");
            refresh();
            return;
        }

        // Header
        attron(A_BOLD);
        mvprintw(0, (term_width_ - 17) / 2, "SPECTRUM ANALYZER");
        attroff(A_BOLD);
        mvhline(1, 0, ACS_HLINE, term_width_);

        // Calculate bar width and spacing
        const auto num_bands = data.magnitudes.size();
        if (num_bands == 0) {
            mvprintw(header_lines + viz_height / 2, (term_width_ - 20) / 2, "Waiting for audio...");
            refresh();
            return;
        }

        // Determine bar layout
        const auto total_bar_space = static_cast<std::size_t>(viz_width);
        int bar_width = static_cast<int>(std::max(std::size_t{1}, total_bar_space / num_bands));
        int gap = 0;

        // Add gaps if we have room
        if (bar_width >= 3) {
            gap = 1;
            const auto usable_space = total_bar_space - (num_bands - 1);
            bar_width = static_cast<int>(std::max(std::size_t{1}, usable_space / num_bands));
        }

        // Render bars
        const int base_y = header_lines + viz_height - 1;
        int x = 1;  // Start with margin

        for (std::size_t i = 0; i < num_bands && x + bar_width <= term_width_ - 1; ++i) {
            const float magnitude = std::clamp(data.magnitudes[i], 0.0f, 1.0f);
            const float peak = std::clamp(data.peaks[i], 0.0f, 1.0f);

            const int bar_height = static_cast<int>(magnitude * static_cast<float>(viz_height - 1));
            const int peak_y = static_cast<int>(peak * static_cast<float>(viz_height - 1));

            // Draw bar from bottom up
            for (int y = 0; y < bar_height; ++y) {
                // Color based on height (gradient from blue to red)
                int color_pair = 1;
                if (has_color_) {
                    const float height_ratio =
                        static_cast<float>(y) / static_cast<float>(viz_height - 1);
                    if (height_ratio > 0.9f)
                        color_pair = 5;  // Red (loud)
                    else if (height_ratio > 0.7f)
                        color_pair = 4;  // Yellow
                    else if (height_ratio > 0.5f)
                        color_pair = 3;  // Green
                    else if (height_ratio > 0.3f)
                        color_pair = 2;  // Cyan
                    else
                        color_pair = 1;  // Blue (quiet)

                    attron(COLOR_PAIR(color_pair));
                }

                // Draw bar segment
                for (int bx = 0; bx < bar_width; ++bx) {
                    mvaddch(base_y - y, x + bx, ACS_BLOCK);
                }

                if (has_color_) {
                    attroff(COLOR_PAIR(color_pair));
                }
            }

            // Draw peak indicator
            if (peak_y > bar_height && peak_y < viz_height) {
                if (has_color_) {
                    attron(COLOR_PAIR(6) | A_BOLD);
                }
                for (int bx = 0; bx < bar_width; ++bx) {
                    mvaddch(base_y - peak_y, x + bx, ACS_HLINE);
                }
                if (has_color_) {
                    attroff(COLOR_PAIR(6) | A_BOLD);
                }
            }

            x += bar_width + gap;
        }

        // Footer separator
        mvhline(term_height_ - footer_lines, 0, ACS_HLINE, term_width_);

        // Footer info
        mvprintw(term_height_ - 1, 1, "RMS: %.2f  Peak: %.2f  Captured: %lluk  Overruns: %llu",
                 static_cast<double>(data.rms_level), static_cast<double>(data.peak_level),
                 static_cast<unsigned long long>(stats.frames_captured / 1000),
                 static_cast<unsigned long long>(stats.overruns));

        mvprintw(term_height_ - 1, term_width_ - 15, "[q] Quit");

        refresh();
    }

    bool running_;
    bool has_color_ = false;
    int term_width_ = 0;
    int term_height_ = 0;
};

}  // namespace audiovis

// Global renderer pointer for signal handling
static audiovis::TerminalRenderer* g_renderer = nullptr;

static void signal_handler(int /*signum*/) {
    if (g_renderer != nullptr) {
        g_renderer->stop();
    }
}

int main() {
    try {
        // Configure analyzer with reasonable defaults
        audiovis::AudioConfig audio_cfg{
            .sample_rate = 48000, .buffer_frames = 512, .channels = 1, .ring_buffer_seconds = 0.5f};

        audiovis::FFTConfig fft_cfg{.fft_size = 2048,
                                    .window = audiovis::WindowFunction::Hann,
                                    .use_magnitude_db = true,
                                    .db_floor = -60.0f,
                                    .db_ceiling = 0.0f};

        audiovis::AnalyzerConfig analyzer_cfg{.num_bands = 64,
                                              .min_frequency = 20.0f,
                                              .max_frequency = 16000.0f,
                                              .smoothing_factor = 0.6f,
                                              .peak_decay_rate = 0.92f,
                                              .logarithmic_frequency = true};

        audiovis::SpectrumAnalyzer analyzer{audio_cfg, fft_cfg, analyzer_cfg};
        audiovis::TerminalRenderer renderer;

        // Set up signal handling for clean shutdown
        g_renderer = &renderer;
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        // Run until user quits
        renderer.run(analyzer);

        g_renderer = nullptr;
        return 0;

    } catch (const std::exception& e) {
        // Ensure terminal is restored before printing error
        endwin();
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
