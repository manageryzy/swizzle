// Minimal stub so cute/util/debug.hpp can compile host-only.
// We only need the symbols to *parse* — CUTE's error macros are defined but
// never invoked in the verify harness.
#pragma once

typedef int cudaError_t;
enum { cudaSuccess = 0 };

inline const char* cudaGetErrorName(cudaError_t) { return "stub"; }
inline const char* cudaGetErrorString(cudaError_t) { return "stub"; }
inline cudaError_t cudaPeekAtLastError() { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }

#define __CUDA_ARCH_LIST__
