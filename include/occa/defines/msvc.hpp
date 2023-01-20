#ifndef OCCA_DEFINES_MSVC_HEADER
#define OCCA_DEFINES_MSVC_HEADER

// NBN: adapted from compiledDefines.hpp
// FIXME: #defines appropriate for local win64 build
// FIXME: user must edit for current system
// TODO: enable local generation with cmake?

#ifndef OCCA_LINUX_OS
#  define OCCA_LINUX_OS 1
#endif

#ifndef OCCA_MACOS_OS
#  define OCCA_MACOS_OS 2
#endif

#ifndef OCCA_WINDOWS_OS
#  define OCCA_WINDOWS_OS 4
#endif

#ifndef OCCA_WINUX_OS
#  define OCCA_WINUX_OS (OCCA_LINUX_OS | OCCA_WINDOWS_OS)
#endif

#define OCCA_OS             OCCA_WINDOWS_OS
#define OCCA_USING_VS       1
#define OCCA_VS_VERSION     _MSC_VER
#define OCCA_UNSAFE         0

#define OCCA_MPI_ENABLED    1
#define OCCA_OPENMP_ENABLED 1
#define OCCA_CUDA_ENABLED   1
#define OCCA_DPCPP_ENABLED  0
#define OCCA_HIP_ENABLED    0
#define OCCA_OPENCL_ENABLED 0
#define OCCA_METAL_ENABLED  0

// TODO: select appropriate intrinsics
#define __AVX__
#define __SSE4_2__
#define __SSE4_1__
#define __SSE3__
#define __SSE2__

// TODO: define appropriate paths
#define OCCA_BUILD_DIR     "D:/TW/occa"
#define OCCA_SOURCE_DIR    "D:/TW/occa"

#ifdef NDEBUG
#  define OCCA_DEBUG_ENABLED 0
#else
#  define OCCA_DEBUG_ENABLED 1
#endif

#endif  // OCCA_DEFINES_MSVC_HEADER
