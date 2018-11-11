#ifndef OCCA_C_DEFINES_HEADER
#define OCCA_C_DEFINES_HEADER

#include <occa/defines.hpp>

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
#  define OCCA_RFUNC
#  define OCCA_LFUNC
#else
#  define OCCA_RFUNC __stdcall
#  ifdef OCCA_C_EXPORTS
//   #define OCCA_LFUNC __declspec(dllexport)
#    define OCCA_LFUNC
#  else
//   #define OCCA_LFUNC __declspec(dllimport)
#    define OCCA_LFUNC
#  endif
#endif

#define OCCA_C_TYPE_MAGIC_HEADER     0x3030CE64
#define OCCA_C_TYPE_UNDEFINED_HEADER 0x770C2D18

#endif
