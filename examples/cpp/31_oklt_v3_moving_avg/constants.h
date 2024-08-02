#pragma once

constexpr const int THREADS_PER_BLOCK = 1024;
//INFO: it's not possible to setup dynamicaly extern @shared array for CUDA
constexpr const int WINDOW_SIZE = 16;
