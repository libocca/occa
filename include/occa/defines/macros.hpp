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

#include "codegen/macros.hpp_codegen"

#endif // OCCA_DISABLE_VARIADIC_MACROS

#endif
