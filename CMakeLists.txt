cmake_minimum_required(VERSION 3.20)

project(audiovis
    VERSION 0.1.0
    DESCRIPTION "Real-time audio visualizer"
    LANGUAGES CXX
)

# Require C++20 for designated initializers, concepts, and std::atomic improvements
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json for tooling (clangd, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type defaults to Release with debug info
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(AUDIOVIS_BUILD_TESTS "Build unit tests" ON)
option(AUDIOVIS_ENABLE_SANITIZERS "Enable ASan/UBSan in Debug builds" ON)
option(AUDIOVIS_USE_TERMINAL "Build terminal visualizer (otherwise SDL2)" ON)

# -----------------------------------------------------------------------------
# Compiler warnings - be strict
# -----------------------------------------------------------------------------
add_library(audiovis_warnings INTERFACE)
target_compile_options(audiovis_warnings INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall -Wextra -Wpedantic -Werror
        -Wconversion -Wsign-conversion
        -Wcast-qual -Wcast-align
        -Wshadow -Wold-style-cast
        -Wnull-dereference
        -Wdouble-promotion
        -Wformat=2
    >
    $<$<CXX_COMPILER_ID:GNU>:
        -Wduplicated-cond -Wduplicated-branches
        -Wlogical-op -Wuseless-cast
    >
)

# -----------------------------------------------------------------------------
# Sanitizers for debug builds
# -----------------------------------------------------------------------------
add_library(audiovis_sanitizers INTERFACE)
if(AUDIOVIS_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(audiovis_sanitizers INTERFACE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )
    target_link_options(audiovis_sanitizers INTERFACE
        -fsanitize=address,undefined
    )
endif()

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# PortAudio for cross-platform audio capture
find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)

# FFTW3 for spectrum analysis
pkg_check_modules(FFTW3 REQUIRED fftw3f)  # float version

# Terminal UI or SDL2
if(AUDIOVIS_USE_TERMINAL)
    pkg_check_modules(NCURSES REQUIRED ncursesw)
else()
    find_package(SDL2 REQUIRED)
endif()

# -----------------------------------------------------------------------------
# Core library
# -----------------------------------------------------------------------------
add_library(audiovis_core STATIC
    src/ring_buffer.cpp
    src/audio_capture.cpp
    src/fft_processor.cpp
    src/spectrum_analyzer.cpp
)

target_include_directories(audiovis_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PORTAUDIO_INCLUDE_DIRS}
        ${FFTW3_INCLUDE_DIRS}
)

target_link_libraries(audiovis_core
    PRIVATE
        audiovis_warnings
        audiovis_sanitizers
        ${PORTAUDIO_LIBRARIES}
        ${FFTW3_LIBRARIES}
)

target_compile_definitions(audiovis_core
    PRIVATE
        $<$<BOOL:${AUDIOVIS_USE_TERMINAL}>:AUDIOVIS_USE_TERMINAL>
)

# -----------------------------------------------------------------------------
# Executable
# -----------------------------------------------------------------------------
add_executable(audiovis src/main.cpp)

if(AUDIOVIS_USE_TERMINAL)
    target_sources(audiovis PRIVATE src/terminal_renderer.cpp)
    target_include_directories(audiovis PRIVATE ${NCURSES_INCLUDE_DIRS})
    target_link_libraries(audiovis PRIVATE ${NCURSES_LIBRARIES})
else()
    target_sources(audiovis PRIVATE src/sdl_renderer.cpp)
    target_link_libraries(audiovis PRIVATE SDL2::SDL2)
endif()

target_link_libraries(audiovis
    PRIVATE
        audiovis_core
        audiovis_warnings
        audiovis_sanitizers
)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
if(AUDIOVIS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
install(TARGETS audiovis RUNTIME DESTINATION bin)
