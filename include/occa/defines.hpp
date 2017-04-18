/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#ifndef OCCA_DEFINES_HEADER
#define OCCA_DEFINES_HEADER

#include "occa/defines/compiledDefines.hpp"

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
#    define OCCA_OS OCCA_OSX_OS
#  else
#    define OCCA_OS OCCA_LINUX_OS
#  endif
#endif

#if OCCA_USING_VS
#  define OCCA_VS_VERSION _MSC_VER
#  include "occa/defines/visualStudio.hpp"
#endif

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

#if   (OCCA_OS == OCCA_LINUX_OS) || (OCCA_OS == OCCA_OSX_OS)
#  define OCCA_INLINE inline __attribute__ ((always_inline))
#elif (OCCA_OS == OCCA_WINDOWS_OS)
#  define OCCA_INLINE __forceinline
#endif

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

//---[ Checks and Info ]----------------
#define OCCA_TEMPLATE_CHECK(checkFunction, expr, filename, function, line, message) \
  do {                                                                  \
    const bool isOK = (bool) (expr);                                    \
    if (!isOK) {                                                        \
      std::stringstream _check_ss;                                      \
      _check_ss << message;                                             \
      checkFunction(filename, function, line, _check_ss.str());         \
    }                                                                   \
  } while(0)

#define OCCA_ERROR3(expr, filename, function, line, message) OCCA_TEMPLATE_CHECK(occa::error, expr, filename, function, line, message)
#define OCCA_ERROR2(expr, filename, function, line, message) OCCA_ERROR3(expr, filename, function, line, message)
#define OCCA_ERROR(message, expr) OCCA_ERROR2(expr, __FILE__, __PRETTY_FUNCTION__, __LINE__, message)

#define OCCA_WARNING3(expr, filename, function, line, message) OCCA_TEMPLATE_CHECK(occa::warn, expr, filename, function, line, message)
#define OCCA_WARNING2(expr, filename, function, line, message) OCCA_WARNING3(expr, filename, function, line, message)
#define OCCA_WARNING(message, expr) OCCA_WARNING2(expr, __FILE__, __PRETTY_FUNCTION__, __LINE__, message)

#define OCCA_FORCE_ERROR(message) OCCA_ERROR(message, false)
#define OCCA_FORCE_WARNING(message)  OCCA_WARNING(message, false)

#define OCCA_DEFAULT_MEM_BYTE_ALIGN 32

//---[ Compiler ]-------------
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
#if defined(__clang__)
#  define OCCA_COMPILED_WITH OCCA_LLVM_COMPILER
#elif defined(__ICC) || defined(__INTEL_COMPILER)
#  define OCCA_COMPILED_WITH OCCA_INTEL_COMPILER
#elif defined(__GNUC__) || defined(__GNUG__)
#  define OCCA_COMPILED_WITH OCCA_GNU_COMPILER
#elif defined(__HP_cc) || defined(__HP_aCC)
#  define OCCA_COMPILED_WITH OCCA_HP_COMPILER
#elif defined(__IBMC__) || defined(__IBMCPP__)
#  define OCCA_COMPILED_WITH OCCA_IBM_COMPILER
#elif defined(__PGI)
#  define OCCA_COMPILED_WITH OCCA_PGI_COMPILER
#elif defined(_CRAYC)
#  define OCCA_COMPILED_WITH OCCA_CRAY_COMPILER
#elif defined(__PATHSCALE__) || defined(__PATHCC__)
#  define OCCA_COMPILED_WITH OCCA_PATHSCALE_COMPILER
#elif defined(_MSC_VER)
#  define OCCA_COMPILED_WITH OCCA_VS_COMPILER
#else
#  define OCCA_COMPILED_WITH OCCA_UNKNOWN_COMPILER
#endif
#endif

//---[ Vectorization ]--------
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
#if OCCA_MIC
#  define OCCA_VECTOR_SET "MIC AVX-512"
#elif OCCA_AVX2
#  define OCCA_VECTOR_SET "AVX2"
#elif OCCA_AVX
#  define OCCA_VECTOR_SET "AVX"
#elif OCCA_SSE4
#  define OCCA_VECTOR_SET "SSE4"
#elif OCCA_SSE3
#  define OCCA_VECTOR_SET "SSE3"
#elif OCCA_SSE2
#  define OCCA_VECTOR_SET "SSE2"
#elif OCCA_SSE
#  define OCCA_VECTOR_SET "SSE"
#elif OCCA_MMX
#  define OCCA_VECTOR_SET "MMX"
#else
#  define OCCA_VECTOR_SET "[Vector Instruction Set Not Found]"
#endif
#endif

#ifndef OCCA_SIMD_WIDTH
#if   OCCA_MIC
#  define OCCA_SIMD_WIDTH 16
#elif OCCA_AVX | OCCA_AVX2
#  define OCCA_SIMD_WIDTH 8
#elif OCCA_SSE | OCCA_SSE2 | OCCA_SSE3 | OCCA_SSE4
#  define OCCA_SIMD_WIDTH 4
#elif OCCA_MMX
#  define OCCA_SIMD_WIDTH 2
#else
#  define OCCA_SIMD_WIDTH 1
#endif
#endif
//============================

#define OCCA_MAX_ARGS 50
//======================================


//---[ OpenCL ]-------------------------
#define OCCA_OPENCL_TEMPLATE_CHECK(checkFunction, expr, filename, function, line, message) \
  do {                                                                  \
    cl_int _clErrorCode = expr;                                         \
    if (_clErrorCode) {                                                 \
      std::stringstream _check_ss;                                      \
      _check_ss << message;                                             \
      checkFunction(_clErrorCode, filename, function, line, _check_ss.str()); \
    }                                                                   \
  } while(0)

#define OCCA_OPENCL_ERROR3(expr, filename, function, line, message) OCCA_OPENCL_TEMPLATE_CHECK(occa::opencl::error, expr, filename, function, line, message)
#define OCCA_OPENCL_ERROR2(expr, filename, function, line, message) OCCA_OPENCL_ERROR3(expr, filename, function, line, message)
#define OCCA_OPENCL_ERROR(message, expr) OCCA_OPENCL_ERROR2(expr, __FILE__, __PRETTY_FUNCTION__, __LINE__, message)

#define OCCA_OPENCL_WARNING3(expr, filename, function, line, message) OCCA_OPENCL_TEMPLATE_CHECK(occa::opencl::warn, expr, filename, function, line, message)
#define OCCA_OPENCL_WARNING2(expr, filename, function, line, message) OCCA_OPENCL_WARNING3(expr, filename, function, line, message)
#define OCCA_OPENCL_WARNING(message, expr) OCCA_OPENCL_WARNING2(expr, __FILE__, __PRETTY_FUNCTION__, __LINE__, message)
//======================================


//---[ CUDA ]---------------------------
#define OCCA_CUDA_TEMPLATE_CHECK(checkFunction, expr, filename, function, line, message) \
  do {                                                                  \
    CUresult _cudaErrorCode = expr;                                     \
    if (_cudaErrorCode) {                                               \
      std::stringstream _check_ss;                                      \
      _check_ss << message;                                             \
      checkFunction(_cudaErrorCode, filename, function, line, _check_ss.str()); \
    }                                                                   \
  } while(0)

#define OCCA_CUDA_ERROR3(expr, filename, function, line, message) OCCA_CUDA_TEMPLATE_CHECK(occa::cuda::error, expr, filename, function, line, message)
#define OCCA_CUDA_ERROR2(expr, filename, function, line, message) OCCA_CUDA_ERROR3(expr, filename, function, line, message)
#define OCCA_CUDA_ERROR(message, expr) OCCA_CUDA_ERROR2(expr, __FILE__, __PRETTY_FUNCTION__, __LINE__, message)

#define OCCA_CUDA_WARNING3(expr, filename, function, line, message) OCCA_CUDA_TEMPLATE_CHECK(occa::cuda::warn, expr, filename, function, line, message)
#define OCCA_CUDA_WARNING2(expr, filename, function, line, message) OCCA_CUDA_WARNING3(expr, filename, function, line, message)
#define OCCA_CUDA_WARNING(message, expr) OCCA_CUDA_WARNING2(expr, __FILE__, __PRETTY_FUNCTION__, __LINE__, message)
//======================================

#endif
