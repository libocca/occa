#ifndef OCCA_DEFINES_MACROS_HEADER
#define OCCA_DEFINES_MACROS_HEADER

#include <occa/defines/compiledDefines.hpp>

#ifndef __PRETTY_FUNCTION__
#  define __PRETTY_FUNCTION__ __FUNCTION__
#endif

#ifdef __cplusplus
#  define OCCA_START_EXTERN_C extern "C" {
#  define OCCA_END_EXTERN_C   }
#else
#  define OCCA_START_EXTERN_C
#  define OCCA_END_EXTERN_C
#endif

#if   (OCCA_OS == OCCA_LINUX_OS) || (OCCA_OS == OCCA_MACOS_OS)
#  define OCCA_INLINE inline __attribute__ ((always_inline))
#elif (OCCA_OS == OCCA_WINDOWS_OS)
#  define OCCA_INLINE __forceinline
#endif

// Just in case someone wants to run with an older format than C99
#ifndef OCCA_DISABLE_VARIADIC_MACROS

#  define OCCA_ARG_COUNT(...) OCCA_ARG_COUNT2(\
    __VA_ARGS__,                              \
    50, 49, 48, 47, 46, 45, 44, 43, 42, 41,   \
    40, 39, 38, 37, 36, 35, 34, 33, 32, 31,   \
    30, 29, 28, 27, 26, 25, 24, 23, 22, 21,   \
    20, 19, 18, 17, 16, 15, 14, 13, 12, 11,   \
    10, 9, 8, 7, 6, 5, 4, 3, 2, 1,            \
    0)

#  define OCCA_ARG_COUNT2(                          \
  _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,          \
  _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, \
  _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, \
  _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, \
  _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, \
  N,  ...) N

#endif // OCCA_DISABLE_VARIADIC_MACROS

#endif
