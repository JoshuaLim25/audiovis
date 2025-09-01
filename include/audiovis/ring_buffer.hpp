#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>
#include <span>
#include <type_traits>

namespace audiovis {

/// Single-producer single-consumer lock-free ring buffer.
///
/// Designed for real-time audio: the producer (audio callback) writes samples
/// without blocking, and the consumer (visualization thread) reads at its own
/// pace. Uses acquire-release semantics for correct synchronization without
/// full memory barriers.
///
/// Template parameter T should be trivially copyable (typically float for audio).
template <typename T>
    requires std::is_trivially_copyable_v<T>
class RingBuffer {
public:
    /// Constructs a ring buffer with the given capacity.
    /// Capacity is rounded up to the next power of two for efficient modulo.
    explicit RingBuffer(std::size_t min_capacity)
        : capacity_{next_power_of_two(min_capacity)}
        , mask_{capacity_ - 1}
        , buffer_{std::make_unique<T[]>(capacity_)} {
        assert(capacity_ > 0 && (capacity_ & mask_) == 0);
    }

    // Non-copyable, non-movable (atomics don't move safely)
    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;
    RingBuffer(RingBuffer&&) = delete;
    RingBuffer& operator=(RingBuffer&&) = delete;

    /// Returns the buffer capacity (always a power of two).
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    /// Returns the number of elements available for reading.
    /// Safe to call from any thread.
    [[nodiscard]] std::size_t size() const noexcept {
        const auto w = write_pos_.load(std::memory_order_acquire);
        const auto r = read_pos_.load(std::memory_order_acquire);
        return w - r;  // Unsigned arithmetic handles wrap-around
    }

    /// Returns available space for writing.
    [[nodiscard]] std::size_t available() const noexcept {
        return capacity_ - size();
    }

    /// Returns true if the buffer is empty.
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }

    /// Returns true if the buffer is full.
    [[nodiscard]] bool full() const noexcept { return size() == capacity_; }

    // -------------------------------------------------------------------------
    // Producer interface (call from audio thread only)
    // -------------------------------------------------------------------------

    /// Writes a single element. Returns true on success, false if full.
    /// Lock-free and wait-free. Safe for real-time audio callbacks.
    bool try_push(const T& value) noexcept {
        const auto w = write_pos_.load(std::memory_order_relaxed);
        const auto r = read_pos_.load(std::memory_order_acquire);

        if (w - r >= capacity_) {
            return false;  // Buffer full
        }

        buffer_[w & mask_] = value;
        write_pos_.store(w + 1, std::memory_order_release);
        return true;
    }

    /// Writes multiple elements from a span. Returns number of elements written.
    /// May write fewer than requested if buffer fills.
    std::size_t try_push(std::span<const T> data) noexcept {
        const auto w = write_pos_.load(std::memory_order_relaxed);
        const auto r = read_pos_.load(std::memory_order_acquire);
        const auto avail = capacity_ - (w - r);
        const auto to_write = std::min(avail, data.size());

        for (std::size_t i = 0; i < to_write; ++i) {
            buffer_[(w + i) & mask_] = data[i];
        }

        write_pos_.store(w + to_write, std::memory_order_release);
        return to_write;
    }

    /// Overwrites oldest data if buffer is full. Always succeeds.
    /// Use when dropping old samples is preferable to blocking.
    void push_overwrite(const T& value) noexcept {
        const auto w = write_pos_.load(std::memory_order_relaxed);
        buffer_[w & mask_] = value;
        write_pos_.store(w + 1, std::memory_order_release);

        // If we've overrun, advance read pointer
        const auto r = read_pos_.load(std::memory_order_relaxed);
        if (w + 1 - r > capacity_) {
            read_pos_.store(w + 1 - capacity_, std::memory_order_release);
        }
    }

    // -------------------------------------------------------------------------
    // Consumer interface (call from visualization thread only)
    // -------------------------------------------------------------------------

    /// Reads a single element. Returns true on success, false if empty.
    bool try_pop(T& out) noexcept {
        const auto r = read_pos_.load(std::memory_order_relaxed);
        const auto w = write_pos_.load(std::memory_order_acquire);

        if (r >= w) {
            return false;  // Buffer empty
        }

        out = buffer_[r & mask_];
        read_pos_.store(r + 1, std::memory_order_release);
        return true;
    }

    /// Reads multiple elements into a span. Returns number of elements read.
    std::size_t try_pop(std::span<T> out) noexcept {
        const auto r = read_pos_.load(std::memory_order_relaxed);
        const auto w = write_pos_.load(std::memory_order_acquire);
        const auto available = w - r;
        const auto to_read = std::min(available, out.size());

        for (std::size_t i = 0; i < to_read; ++i) {
            out[i] = buffer_[(r + i) & mask_];
        }

        read_pos_.store(r + to_read, std::memory_order_release);
        return to_read;
    }

    /// Peeks at data without consuming it. Copies up to `count` elements.
    /// Returns number of elements copied.
    std::size_t peek(std::span<T> out) const noexcept {
        const auto r = read_pos_.load(std::memory_order_relaxed);
        const auto w = write_pos_.load(std::memory_order_acquire);
        const auto available = w - r;
        const auto to_copy = std::min(available, out.size());

        for (std::size_t i = 0; i < to_copy; ++i) {
            out[i] = buffer_[(r + i) & mask_];
        }

        return to_copy;
    }

    /// Discards up to `count` elements. Returns number discarded.
    std::size_t discard(std::size_t count) noexcept {
        const auto r = read_pos_.load(std::memory_order_relaxed);
        const auto w = write_pos_.load(std::memory_order_acquire);
        const auto available = w - r;
        const auto to_discard = std::min(available, count);

        read_pos_.store(r + to_discard, std::memory_order_release);
        return to_discard;
    }

    /// Clears all data. Safe to call from consumer thread.
    void clear() noexcept {
        const auto w = write_pos_.load(std::memory_order_acquire);
        read_pos_.store(w, std::memory_order_release);
    }

private:
    static constexpr std::size_t next_power_of_two(std::size_t n) noexcept {
        if (n == 0) return 1;
        --n;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        return n + 1;
    }

    const std::size_t capacity_;
    const std::size_t mask_;
    std::unique_ptr<T[]> buffer_;

    // Cache-line padding to prevent false sharing between producer and consumer.
    // Using explicit 64-byte alignment (common cache line size on x86-64 and ARM64)
    // rather than std::hardware_destructive_interference_size, which GCC warns about
    // due to ABI instability across compiler versions and tuning flags.
    static constexpr std::size_t kCacheLineSize = 64;

    alignas(kCacheLineSize) std::atomic<std::size_t> write_pos_{0};
    alignas(kCacheLineSize) std::atomic<std::size_t> read_pos_{0};
};

}  // namespace audiovis
