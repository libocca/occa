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
#include "../parserUtils.hpp"

void testPreprocessor();
void testLoopExtraction();
void testGlobalConst();
void testDeviceFunctions();
void testKernelAnnotation();
void testSharedAnnotation();
void testBarriers();

int main(const int argc, const char **argv) {
  parser.addAttribute<dummy>();
  parser.addAttribute<attributes::kernel>();
  parser.addAttribute<attributes::outer>();
  parser.addAttribute<attributes::inner>();
  parser.addAttribute<attributes::shared>();
  parser.addAttribute<attributes::exclusive>();

  testPreprocessor();
  testLoopExtraction();
  testGlobalConst();
  testDeviceFunctions();
  testKernelAnnotation();
  testSharedAnnotation();
  testBarriers();

  return 0;
}

//---[ Preprocessor ]-------------------
void testPreprocessor() {
  // #define restrict __restrict__
}
//======================================

//---[ Loops ]--------------------------
void testLoopExtraction() {
  // SPLIT LOOPS!!
}
//======================================

//---[ Constant ]-----------------------
void testGlobalConst() {
  // Global const -> __constant__
}
//======================================

//---[ Device Functions ]---------------
void testDeviceFunctions() {
  // Non-@kernel function [N/A, __device__]
}
//======================================

//---[ Kernel ]-------------------------
void testKernelAnnotation() {
  // @kernel -> extern "C" __global__
}
//======================================

//---[ Shared ]-------------------------
void testSharedAnnotation() {
  // @shared -> __shared__
}
//======================================

//---[ Barriers ]-----------------------
void testBarriers() {
  // Add barriers __syncthreads()
}
//======================================
