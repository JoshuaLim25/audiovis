include(FetchContent)

# Fetch GoogleTest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Ring buffer tests
add_executable(test_ring_buffer test_ring_buffer.cpp)
target_link_libraries(test_ring_buffer
    PRIVATE
        audiovis_core
        audiovis_warnings
        GTest::gtest_main
)
add_test(NAME RingBufferTests COMMAND test_ring_buffer)

# FFT processor tests
add_executable(test_fft_processor test_fft_processor.cpp)
target_link_libraries(test_fft_processor
    PRIVATE
        audiovis_core
        audiovis_warnings
        GTest::gtest_main
)
add_test(NAME FFTProcessorTests COMMAND test_fft_processor)
