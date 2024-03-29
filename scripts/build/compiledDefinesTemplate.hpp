#ifndef OCCA_DEFINES_COMPILEDDEFINES_HEADER
#define OCCA_DEFINES_COMPILEDDEFINES_HEADER

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

#define OCCA_OS             @@OCCA_OS@@
#define OCCA_USING_VS       @@OCCA_USING_VS@@
#define OCCA_UNSAFE         @@OCCA_UNSAFE@@

#define OCCA_OPENMP_ENABLED @@OCCA_OPENMP_ENABLED@@
#define OCCA_CUDA_ENABLED   @@OCCA_CUDA_ENABLED@@
#define OCCA_HIP_ENABLED    @@OCCA_HIP_ENABLED@@
#define OCCA_OPENCL_ENABLED @@OCCA_OPENCL_ENABLED@@
#define OCCA_METAL_ENABLED  @@OCCA_METAL_ENABLED@@
#define OCCA_DPCPP_ENABLED @@OCCA_DPCPP_ENABLED@@

#define OCCA_THREAD_SHARABLE_ENABLED @@OCCA_THREAD_SHARABLE_ENABLED@@
#define OCCA_MAX_ARGS @@OCCA_MAX_ARGS@@

#define OCCA_BUILD_DIR     "@@OCCA_BUILD_DIR@@"

#endif
