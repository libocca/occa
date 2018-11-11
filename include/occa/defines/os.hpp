#ifndef OCCA_DEFINES_OS_HEADER
#define OCCA_DEFINES_OS_HEADER

#include <occa/defines/compiledDefines.hpp>

#ifndef OCCA_USING_VS
#  ifdef _MSC_VER
#    define OCCA_USING_VS 1
#    define OCCA_OS OCCA_WINDOWS_OS
#  else
#    define OCCA_USING_VS 0
#  endif
#endif

#ifndef OCCA_OS
#  if defined(WIN32) || defined(WIN64)
#    if OCCA_USING_VS
#      define OCCA_OS OCCA_WINDOWS_OS
#    else
#      define OCCA_OS OCCA_WINUX_OS
#    endif
#  elif __APPLE__
#    define OCCA_OS OCCA_MACOS_OS
#  else
#    define OCCA_OS OCCA_LINUX_OS
#  endif
#endif

#endif
