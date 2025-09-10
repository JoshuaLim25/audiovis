// Main entry point for audiovis.
//
// This file exists to allow CMake to conditionally compile different
// renderers (terminal vs SDL2) while keeping the main() function in
// the appropriate renderer source file.
//
// When AUDIOVIS_USE_TERMINAL is defined, main() is in terminal_renderer.cpp.
// Otherwise, main() is in sdl_renderer.cpp.
//
// This file intentionally left mostly empty - it serves as a placeholder
// for potential future CLI argument parsing that's shared across renderers.

#ifndef AUDIOVIS_USE_TERMINAL
// SDL2 renderer will define main() - this file is not compiled in that case
#endif
