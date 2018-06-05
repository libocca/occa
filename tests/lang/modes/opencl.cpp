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
#define OCCA_TEST_PARSER_TYPE okl::openclParser

#include "occa/lang/modes/opencl.hpp"
#include "../parserUtils.hpp"

void testPragma();
void testLoopExtraction();
void testGlobalConst();
void testKernelAnnotation();
void testKernelArgs();
void testSharedAnnotation();
void testBarriers();
void testSource();

int main(const int argc, const char **argv) {
  parser.settings["okl/validate"] = false;
  testPragma();

  parser.settings["okl/validate"] = true;
  testLoopExtraction();
  testGlobalConst();
  testKernelAnnotation();
  testKernelArgs();
  testSharedAnnotation();
  testBarriers();
  testSource();

  return 0;
}

//---[ Pragma ]-------------------------
void testPragma() {
  parseSource("");
  OCCA_ASSERT_EQUAL(1,
                    parser.root.size());

  OCCA_ASSERT_EQUAL("OPENCL EXTENSION cl_khr_fp64 : enable\n",
                    parser.root[0]
                    ->to<pragmaStatement>()
                    .value());

  parser.settings["opencl/extensions/cl_khr_fp64"] = false;
  parseSource("");
  OCCA_ASSERT_EQUAL(0,
                    parser.root.size());


  parser.settings["opencl/extensions/foobar"] = true;
  parseSource("");
  OCCA_ASSERT_EQUAL(1,
                    parser.root.size());

  OCCA_ASSERT_EQUAL("OPENCL EXTENSION foobar : enable\n",
                    parser.root[0]
                    ->to<pragmaStatement>()
                    .value());

  parser.settings["opencl/extensions/foobar"] = false;
  parser.settings["opencl/extensions/cl_khr_fp64"] = true;
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

void testSource() {
  // TODO:
  //   @exclusive ->
  //     - std::vector<value>
  //     - vec.reserve(loopIterations)
  //     - Add iterator index to inner-most @inner loop
  parseSource(
    "const int var[10];\n"
    "@kernel void foo(int * restrict arg, const int bar) {\n"
    "  for (int o1 = 0; o1 < O1; ++o1; @outer) {\n"
    "    for (int o0 = 0; o0 < O0; ++o0; @outer) {\n"
    "      @shared int shr[3];\n"
    "      @exclusive int excl;\n"
    "      if (true) {\n"
    "        for (int i1 = 10; i1 < (I1 + 4); i1 += 3; @inner) {\n"
    "          for (int i0 = 0; i0 < I0; ++i0; @inner) {\n"
    "            for (;;) {\n"
    "               excl = i0;\n"
    "            }\n"
    "            for (;;) {\n"
    "               excl = i0;\n"
    "            }\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n"
  );
}
