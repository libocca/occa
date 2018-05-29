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

void testPragma();
void testLoopExtraction();
void testGlobalConst();
void testKernelAnnotation();
void testKernelArgs();
void testSharedAnnotation();
void testBarriers();

int main(const int argc, const char **argv) {
  parser.addAttribute<dummy>();
  parser.addAttribute<attributes::kernel>();
  parser.addAttribute<attributes::outer>();
  parser.addAttribute<attributes::inner>();
  parser.addAttribute<attributes::shared>();
  parser.addAttribute<attributes::exclusive>();

  testPragma();
  testLoopExtraction();
  testGlobalConst();
  testKernelAnnotation();
  testKernelArgs();
  testSharedAnnotation();
  testBarriers();

  return 0;
}

//---[ Pragma ]-------------------------
void testPragma() {
  // #pragma OPENCL EXTENSION cl_khr_fp64 : enable
}
//======================================

//---[ Loops ]--------------------------
void testLoopExtraction() {
  // SPLIT LOOPS!!
}
//======================================

//---[ Constant ]-----------------------
void testGlobalConst() {
  // Global const -> __constant
}
//======================================

//---[ Kernel ]-------------------------
void testKernelAnnotation() {
  // @kernel -> __kernel
}
//======================================

//---[ Kernel Args ]--------------------
void testKernelArgs() {
  // @kernel arg -> __global
}
//======================================

//---[ Shared ]-------------------------
void testSharedAnnotation() {
  // @shared -> __local
}
//======================================

//---[ Barriers ]-----------------------
void testBarriers() {
  // Add barriers barrier(CLK_LOCAL_MEM_FENCE)
}
//======================================
