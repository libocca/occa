#ifndef OCCA_DEFINES_ARCH_HEADER
#define OCCA_DEFINES_ARCH_HEADER

#include <occa/defines/compiledDefines.hpp>


//---[ Architecture ]-------------------
#if defined __arm__
#  define OCCA_ARM 1
#else
#  define OCCA_ARM 0
#endif

#if defined(__x86_64__) || defined(_M_X64) // 64 Bit
#  define OCCA_64_BIT 1
#  define OCCA_32_BIT 0
#elif defined(__i386) || defined(_M_IX86) // 32 Bit
#  define OCCA_64_BIT 0
#  define OCCA_32_BIT 1
#elif defined(__ia64) || defined(__itanium__) || defined(_A_IA64) // Itanium
#  define OCCA_64_BIT 1
#  define OCCA_32_BIT 0
#endif
//======================================


//---[ Compiler ]-----------------------
#define OCCA_GNU_COMPILER       (1 << 0)
#define OCCA_LLVM_COMPILER      (1 << 1)
#define OCCA_INTEL_COMPILER     (1 << 2)
#define OCCA_PATHSCALE_COMPILER (1 << 3)
#define OCCA_IBM_COMPILER       (1 << 4)
#define OCCA_PGI_COMPILER       (1 << 5)
#define OCCA_HP_COMPILER        (1 << 6)
#define OCCA_VS_COMPILER        (1 << 7)
#define OCCA_CRAY_COMPILER      (1 << 8)
#define OCCA_UNKNOWN_COMPILER   (1 << 9)

#ifndef OCCA_COMPILED_WITH
#  if defined(__clang__)
#    define OCCA_COMPILED_WITH OCCA_LLVM_COMPILER
#  elif defined(__ICC) || defined(__INTEL_COMPILER)
#    define OCCA_COMPILED_WITH OCCA_INTEL_COMPILER
#  elif defined(__GNUC__) || defined(__GNUG__)
#    define OCCA_COMPILED_WITH OCCA_GNU_COMPILER
#  elif defined(__HP_cc) || defined(__HP_aCC)
#    define OCCA_COMPILED_WITH OCCA_HP_COMPILER
#  elif defined(__IBMC__) || defined(__IBMCPP__)
#    define OCCA_COMPILED_WITH OCCA_IBM_COMPILER
#  elif defined(__PGI)
#    define OCCA_COMPILED_WITH OCCA_PGI_COMPILER
#  elif defined(_CRAYC)
#    define OCCA_COMPILED_WITH OCCA_CRAY_COMPILER
#  elif defined(__PATHSCALE__) || defined(__PATHCC__)
#    define OCCA_COMPILED_WITH OCCA_PATHSCALE_COMPILER
#  elif defined(_MSC_VER)
#    define OCCA_COMPILED_WITH OCCA_VS_COMPILER
#  else
#    define OCCA_COMPILED_WITH OCCA_UNKNOWN_COMPILER
#  endif
#endif
//======================================


//---[ Vectorization ]------------------
#ifdef __MIC__
#  define OCCA_MIC 1
#else
#  define OCCA_MIC 0
#endif

#ifdef __AVX2__
#  define OCCA_AVX2 1
#else
#  define OCCA_AVX2 0
#endif

#ifdef __AVX__
#  define OCCA_AVX 1
#else
#  define OCCA_AVX 0
#endif

#ifdef __SSE4_2__
#  define OCCA_SSE4_2 1
#else
#  define OCCA_SSE4_2 0
#endif

#ifdef __SSE4_1__
#  define OCCA_SSE4_1 1
#else
#  define OCCA_SSE4_1 0
#endif

#ifndef OCCA_SSE4
#  if OCCA_SSE4_1 || OCCA_SSE4_2
#    define OCCA_SSE4 1
#  else
#    define OCCA_SSE4 0
#  endif
#endif

#ifdef __SSE3__
#  define OCCA_SSE3 1
#else
#  define OCCA_SSE3 0
#endif

#ifndef OCCA_SSE3
#  ifdef __SSE3__
#    define OCCA_SSE3 1
#  else
#    define OCCA_SSE3 0
#  endif
#endif

#ifndef OCCA_SSE2
#ifdef __SSE2__
#  define OCCA_SSE2 1
#else
#  define OCCA_SSE2 0
#endif
#endif

#ifndef OCCA_SSE
#ifdef __SSE__
#  define OCCA_SSE 1
#else
#  define OCCA_SSE 0
#endif
#endif

#ifndef OCCA_MMX
#ifdef __MMX__
#  define OCCA_MMX 1
#else
#  define OCCA_MMX 0
#endif
#endif

#ifndef OCCA_VECTOR_SET
#  if OCCA_MIC
#    define OCCA_VECTOR_SET "MIC AVX-512"
#  elif OCCA_AVX2
#    define OCCA_VECTOR_SET "AVX2"
#  elif OCCA_AVX
#    define OCCA_VECTOR_SET "AVX"
#  elif OCCA_SSE4
#    define OCCA_VECTOR_SET "SSE4"
#  elif OCCA_SSE3
#    define OCCA_VECTOR_SET "SSE3"
#  elif OCCA_SSE2
#    define OCCA_VECTOR_SET "SSE2"
#  elif OCCA_SSE
#    define OCCA_VECTOR_SET "SSE"
#  elif OCCA_MMX
#    define OCCA_VECTOR_SET "MMX"
#  else
#    define OCCA_VECTOR_SET "N/A"
#  endif
#endif

#ifndef OCCA_SIMD_WIDTH
#  if   OCCA_MIC
#    define OCCA_SIMD_WIDTH 16
#  elif OCCA_AVX | OCCA_AVX2
#    define OCCA_SIMD_WIDTH 8
#  elif OCCA_SSE | OCCA_SSE2 | OCCA_SSE3 | OCCA_SSE4
#    define OCCA_SIMD_WIDTH 4
#  elif OCCA_MMX
#    define OCCA_SIMD_WIDTH 2
#  else
#    define OCCA_SIMD_WIDTH 1
#  endif
#endif
//======================================

#endif
