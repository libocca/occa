#ifndef OCCA_DEFINES_MACROS_HEADER
#define OCCA_DEFINES_MACROS_HEADER

#include <occa/defines/compiledDefines.hpp>

#ifndef __PRETTY_FUNCTION__
#  define __PRETTY_FUNCTION__ __FUNCTION__
#endif

#define OCCA_STRINGIFY2(macro) #macro
#define OCCA_STRINGIFY(macro) OCCA_STRINGIFY2(macro)

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

#endif
