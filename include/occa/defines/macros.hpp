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
