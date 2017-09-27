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

#define OCCA_GNU_VENDOR          0
#define OCCA_LLVM_VENDOR         1
#define OCCA_INTEL_VENDOR        2
#define OCCA_PATHSCALE_VENDOR    3
#define OCCA_IBM_VENDOR          4
#define OCCA_PGI_VENDOR          5
#define OCCA_HP_VENDOR           6
#define OCCA_VISUALSTUDIO_VENDOR 7
#define OCCA_CRAY_VENDOR         8
#define OCCA_NOT_FOUND           9

int main(int argc, char **argv) {

#if defined(__xlc__) || defined(__xlC__) \
  || defined(__IBMC__) || defined(__IBMCPP__) \
  || defined( __ibmxl__)
  return OCCA_IBM_VENDOR;

#elif defined(__ICC) || defined(__INTEL_COMPILER)
  return OCCA_INTEL_VENDOR;

#elif defined(__GNUC__) || defined(__GNUG__)
  return OCCA_GNU_VENDOR;

#elif defined(__HP_cc) || defined(__HP_aCC)
  return OCCA_HP_VENDOR;

#elif defined(__PGI)
  return OCCA_PGI_VENDOR;

#elif defined(_CRAYC)
  return OCCA_CRAY_VENDOR;

#elif defined(__PATHSCALE__) || defined(__PATHCC__)
  return OCCA_PATHSCALE_VENDOR;

#elif defined(_MSC_VER)
  return OCCA_VISUALSTUDIO_VENDOR;

#elif defined(__clang__)
  return OCCA_LLVM_VENDOR;

#else
  return OCCA_NOT_FOUND
#endif
}
