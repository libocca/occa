/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
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

#ifndef OCCA_DEFINES_ERRORS_HEADER
#define OCCA_DEFINES_ERRORS_HEADER

#include <occa/defines/compiledDefines.hpp>


//---[ Checks and Info ]----------------
#define OCCA_TEMPLATE_CHECK_(checkFunction, expr, filename, function, line, message) \
  do {                                                                  \
    const bool isOK = (bool) (expr);                                    \
    if (!isOK) {                                                        \
      std::stringstream _check_ss;                                      \
      _check_ss << message;                                             \
      checkFunction(filename, function, line, _check_ss.str());         \
    }                                                                   \
  } while(0)

#if !OCCA_UNSAFE
#  define OCCA_TEMPLATE_CHECK(a,b,c,d,e,f) OCCA_TEMPLATE_CHECK_(a,b,c,d,e,f)
#else
#  define OCCA_TEMPLATE_CHECK(a,b,c,d,e,f)
#endif

#define OCCA_ERROR3(expr, filename, function, line, message) OCCA_TEMPLATE_CHECK(occa::error, expr, filename, function, line, message)
#define OCCA_ERROR2(expr, filename, function, line, message) OCCA_ERROR3(expr, filename, function, line, message)
#define OCCA_ERROR(message, expr)                            OCCA_ERROR2(expr, __FILE__, __PRETTY_FUNCTION__, __LINE__, message)

#define OCCA_WARNING3(expr, filename, function, line, message) OCCA_TEMPLATE_CHECK(occa::warn, expr, filename, function, line, message)
#define OCCA_WARNING2(expr, filename, function, line, message) OCCA_WARNING3(expr, filename, function, line, message)
#define OCCA_WARNING(message, expr)                            OCCA_WARNING2(expr, __FILE__, __PRETTY_FUNCTION__, __LINE__, message)

#define OCCA_FORCE_ERROR(message)   OCCA_ERROR(message, false)
#define OCCA_FORCE_WARNING(message) OCCA_WARNING(message, false)
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


//---[ HIP ]---------------------------
#define OCCA_HIP_TEMPLATE_CHECK(checkFunction, expr, filename, function, line, message) \
  do {                                                                  \
    hipError_t _hipErrorCode = expr;                                     \
    if (_hipErrorCode) {                                               \
      std::stringstream _check_ss;                                      \
      _check_ss << message;                                             \
      checkFunction(_hipErrorCode, filename, function, line, _check_ss.str()); \
    }                                                                   \
  } while(0)

#define OCCA_HIP_ERROR3(expr, filename, function, line, message) OCCA_HIP_TEMPLATE_CHECK(occa::hip::error, expr, filename, function, line, message)
#define OCCA_HIP_ERROR2(expr, filename, function, line, message) OCCA_HIP_ERROR3(expr, filename, function, line, message)
#define OCCA_HIP_ERROR(message, expr) OCCA_HIP_ERROR2(expr, __FILE__, __PRETTY_FUNCTION__, __LINE__, message)

#define OCCA_HIP_WARNING3(expr, filename, function, line, message) OCCA_HIP_TEMPLATE_CHECK(occa::hip::warn, expr, filename, function, line, message)
#define OCCA_HIP_WARNING2(expr, filename, function, line, message) OCCA_HIP_WARNING3(expr, filename, function, line, message)
#define OCCA_HIP_WARNING(message, expr) OCCA_HIP_WARNING2(expr, __FILE__, __PRETTY_FUNCTION__, __LINE__, message)
//======================================

#endif
