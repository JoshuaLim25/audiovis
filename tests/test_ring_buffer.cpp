#include "audiovis/ring_buffer.hpp"

#include <gtest/gtest.h>

#include <array>
#include <thread>
#include <vector>

namespace audiovis {
namespace {

class RingBufferTest : public ::testing::Test {
protected:
    static constexpr std::size_t kDefaultCapacity = 16;
};

TEST_F(RingBufferTest, ConstructsWithPowerOfTwoCapacity) {
    RingBuffer<float> buf{10};
    EXPECT_EQ(buf.capacity(), 16);  // Rounds up to next power of two
}

TEST_F(RingBufferTest, InitiallyEmpty) {
    RingBuffer<float> buf{kDefaultCapacity};
    EXPECT_TRUE(buf.empty());
    EXPECT_FALSE(buf.full());
    EXPECT_EQ(buf.size(), 0);
    EXPECT_EQ(buf.available(), buf.capacity());
}

TEST_F(RingBufferTest, PushIncrementsSize) {
    RingBuffer<float> buf{kDefaultCapacity};

    EXPECT_TRUE(buf.try_push(1.0f));
    EXPECT_EQ(buf.size(), 1);
    EXPECT_FALSE(buf.empty());

    EXPECT_TRUE(buf.try_push(2.0f));
    EXPECT_EQ(buf.size(), 2);
}

TEST_F(RingBufferTest, PopRetrievesInFIFOOrder) {
    RingBuffer<float> buf{kDefaultCapacity};

    buf.try_push(1.0f);
    buf.try_push(2.0f);
    buf.try_push(3.0f);

    float value = 0.0f;
    EXPECT_TRUE(buf.try_pop(value));
    EXPECT_FLOAT_EQ(value, 1.0f);

    EXPECT_TRUE(buf.try_pop(value));
    EXPECT_FLOAT_EQ(value, 2.0f);

    EXPECT_TRUE(buf.try_pop(value));
    EXPECT_FLOAT_EQ(value, 3.0f);

    EXPECT_TRUE(buf.empty());
}

TEST_F(RingBufferTest, PopFromEmptyReturnsFalse) {
    RingBuffer<float> buf{kDefaultCapacity};
    float value = -1.0f;
    EXPECT_FALSE(buf.try_pop(value));
    EXPECT_FLOAT_EQ(value, -1.0f);  // Unchanged
}

TEST_F(RingBufferTest, PushToFullReturnsFalse) {
    RingBuffer<float> buf{4};  // Capacity will be 4

    EXPECT_TRUE(buf.try_push(1.0f));
    EXPECT_TRUE(buf.try_push(2.0f));
    EXPECT_TRUE(buf.try_push(3.0f));
    EXPECT_TRUE(buf.try_push(4.0f));
    EXPECT_TRUE(buf.full());

    EXPECT_FALSE(buf.try_push(5.0f));  // Buffer full
}

TEST_F(RingBufferTest, SpanPushWritesMultipleElements) {
    RingBuffer<float> buf{kDefaultCapacity};
    std::array<float, 4> data = {1.0f, 2.0f, 3.0f, 4.0f};

    EXPECT_EQ(buf.try_push(data), 4);
    EXPECT_EQ(buf.size(), 4);

    float value = 0.0f;
    buf.try_pop(value);
    EXPECT_FLOAT_EQ(value, 1.0f);
}

TEST_F(RingBufferTest, SpanPopReadsMultipleElements) {
    RingBuffer<float> buf{kDefaultCapacity};

    for (float i = 1.0f; i <= 5.0f; i += 1.0f) {
        buf.try_push(i);
    }

    std::array<float, 3> out{};
    EXPECT_EQ(buf.try_pop(out), 3);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);

    EXPECT_EQ(buf.size(), 2);
}

TEST_F(RingBufferTest, PeekDoesNotConsume) {
    RingBuffer<float> buf{kDefaultCapacity};
    buf.try_push(42.0f);

    std::array<float, 1> out{};
    EXPECT_EQ(buf.peek(out), 1);
    EXPECT_FLOAT_EQ(out[0], 42.0f);
    EXPECT_EQ(buf.size(), 1);  // Still there
}

TEST_F(RingBufferTest, DiscardRemovesElements) {
    RingBuffer<float> buf{kDefaultCapacity};

    for (int i = 0; i < 10; ++i) {
        buf.try_push(static_cast<float>(i));
    }

    EXPECT_EQ(buf.discard(3), 3);
    EXPECT_EQ(buf.size(), 7);

    float value = 0.0f;
    buf.try_pop(value);
    EXPECT_FLOAT_EQ(value, 3.0f);  // First three were discarded
}

TEST_F(RingBufferTest, ClearEmptiesBuffer) {
    RingBuffer<float> buf{kDefaultCapacity};

    for (int i = 0; i < 8; ++i) {
        buf.try_push(static_cast<float>(i));
    }

    buf.clear();
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.size(), 0);
}

TEST_F(RingBufferTest, WrapsAroundCorrectly) {
    RingBuffer<float> buf{4};  // Capacity 4

    // Fill buffer
    for (int i = 0; i < 4; ++i) {
        buf.try_push(static_cast<float>(i));
    }

    // Pop two
    float value = 0.0f;
    buf.try_pop(value);
    buf.try_pop(value);

    // Push two more (these wrap around)
    buf.try_push(10.0f);
    buf.try_push(11.0f);

    // Verify order
    buf.try_pop(value);
    EXPECT_FLOAT_EQ(value, 2.0f);
    buf.try_pop(value);
    EXPECT_FLOAT_EQ(value, 3.0f);
    buf.try_pop(value);
    EXPECT_FLOAT_EQ(value, 10.0f);
    buf.try_pop(value);
    EXPECT_FLOAT_EQ(value, 11.0f);
}

TEST_F(RingBufferTest, PushOverwriteDropsOldData) {
    RingBuffer<float> buf{4};

    for (int i = 0; i < 4; ++i) {
        buf.try_push(static_cast<float>(i));
    }

    // Overwrite - should drop oldest
    buf.push_overwrite(100.0f);

    float value = 0.0f;
    buf.try_pop(value);
    EXPECT_FLOAT_EQ(value, 1.0f);  // 0 was dropped
}

// Stress test for thread safety
TEST_F(RingBufferTest, ConcurrentProducerConsumer) {
    constexpr std::size_t kNumItems = 100000;
    RingBuffer<float> buf{1024};

    std::atomic<bool> done{false};
    std::vector<float> received;
    received.reserve(kNumItems);

    // Consumer thread
    std::thread consumer{[&]() {
        float value = 0.0f;
        while (!done.load(std::memory_order_relaxed) || !buf.empty()) {
            if (buf.try_pop(value)) {
                received.push_back(value);
            }
        }
    }};

    // Producer (main thread)
    for (std::size_t i = 0; i < kNumItems; ++i) {
        while (!buf.try_push(static_cast<float>(i))) {
            // Spin if full
            std::this_thread::yield();
        }
    }

    done.store(true, std::memory_order_relaxed);
    consumer.join();

    // Verify all items received in order
    EXPECT_EQ(received.size(), kNumItems);
    for (std::size_t i = 0; i < received.size(); ++i) {
        EXPECT_FLOAT_EQ(received[i], static_cast<float>(i));
    }
}

}  // namespace
}  // namespace audiovis
