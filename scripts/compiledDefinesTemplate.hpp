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

#ifndef OCCA_DEFINES_COMPILEDDEFINES_HEADER
#define OCCA_DEFINES_COMPILEDDEFINES_HEADER

#ifndef OCCA_LINUX_OS
#  define OCCA_LINUX_OS 1
#endif

#ifndef OCCA_OSX_OS
#  define OCCA_OSX_OS 2
#endif

#ifndef OCCA_WINDOWS_OS
#  define OCCA_WINDOWS_OS 4
#endif

#ifndef OCCA_WINUX_OS
#  define OCCA_WINUX_OS (OCCA_LINUX_OS | OCCA_WINDOWS_OS)
#endif

#define OCCA_OS             @@OCCA_OS@@
#define OCCA_USING_VS       @@OCCA_USING_VS@@
#define OCCA_COMPILED_DIR   @@OCCA_COMPILED_DIR@@

#define OCCA_DEBUG_ENABLED  @@OCCA_DEBUG_ENABLED@@
#define OCCA_CHECK_ENABLED  @@OCCA_CHECK_ENABLED@@

#define OCCA_MPI_ENABLED    @@OCCA_MPI_ENABLED@@
#define OCCA_OPENMP_ENABLED @@OCCA_OPENMP_ENABLED@@
#define OCCA_OPENCL_ENABLED @@OCCA_OPENCL_ENABLED@@
#define OCCA_CUDA_ENABLED   @@OCCA_CUDA_ENABLED@@

#endif
