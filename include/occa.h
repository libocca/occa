#ifndef OCCA_C_HEADER
#define OCCA_C_HEADER

#include <occa/c/base.h>
#include <occa/c/device.h>
#include <occa/c/dtype.h>
#include <occa/c/io.h>
#include <occa/c/json.h>
#include <occa/c/kernel.h>
#include <occa/c/memory.h>
#include <occa/c/uva.h>

// Just in case someone wants to run with an older format than C99
#ifndef OCCA_DISABLE_VARIADIC_MACROS

#  define OCCA_C_RUN_KERNEL3(kernel, N, ...)    \
  occaKernelRunN(kernel, N, __VA_ARGS__)

#  define OCCA_C_RUN_KERNEL2(kernel, N, ...)    \
  OCCA_C_RUN_KERNEL3(kernel, N, __VA_ARGS__)

#  define OCCA_C_RUN_KERNEL1(kernel, N, ...)    \
  OCCA_C_RUN_KERNEL2(kernel, N, __VA_ARGS__)

#  define occaKernelRun(kernel, ...)                                    \
  OCCA_C_RUN_KERNEL1(kernel, OCCA_ARG_COUNT(__VA_ARGS__), __VA_ARGS__)

#endif // OCCA_DISABLE_VARIADIC_MACROS

#endif
