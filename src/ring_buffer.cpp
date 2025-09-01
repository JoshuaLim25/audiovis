// Ring buffer is fully header-only (template class).
// This file exists for potential future non-template utilities.

#include "audiovis/ring_buffer.hpp"

namespace audiovis {

// Explicit instantiation for float to ensure template compiles correctly
template class RingBuffer<float>;

}  // namespace audiovis
