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
#define OCCA_DISABLE_VARIADIC_MACROS

#include <occa.hpp>
#include <occa.h>
#include <occa/c/types.hpp>
#include <occa/tools/testing.hpp>

occaKernel addVectors = occaUndefined;
const std::string addVectorsFile = (
  occa::env::OCCA_DIR + "tests/files/addVectors.okl"
);

void testInit();
void testInfo();
void testRun();

int main(const int argc, const char **argv) {
  addVectors = occaBuildKernel(addVectorsFile.c_str(),
                               "addVectors",
                               occaDefault);

  testInit();
  testInfo();
  testRun();

  occaFree(addVectors);

  return 0;
}

void testInit() {
  occaKernel addVectors2 = occaUndefined;

  ASSERT_TRUE(occaIsUndefined(addVectors2));
  ASSERT_EQ(addVectors2.type,
            OCCA_UNDEFINED);
  ASSERT_FALSE(occaKernelIsInitialized(addVectors2));

  addVectors2 = addVectors;

  ASSERT_FALSE(occaIsUndefined(addVectors2));
  ASSERT_EQ(addVectors2.type,
            OCCA_KERNEL);
  ASSERT_TRUE(occaKernelIsInitialized(addVectors2));
}

void testInfo() {
  occaProperties props = occaKernelGetProperties(addVectors);
  occaType mode = occaPropertiesGet(props, "mode", occaUndefined);
  ASSERT_FALSE(occaIsUndefined(mode));
  ASSERT_EQ((const char*) mode.value.ptr,
            (const char*) "Serial");

  occaDevice device = occaKernelGetDevice(addVectors);
  ASSERT_FALSE(occaIsUndefined(device));
  ASSERT_EQ((const char*) occaDeviceMode(device),
            (const char*) "Serial");

  ASSERT_EQ((const char*) occaKernelName(addVectors),
            (const char*) "addVectors");

  ASSERT_EQ((const char*) occaKernelSourceFilename(addVectors),
            (const char*) addVectorsFile.c_str());

  std::string binaryFilename = occaKernelBinaryFilename(addVectors);
  ASSERT_TRUE(
    occa::startsWith(binaryFilename, occa::io::cachePath())
  );

  occaKernelMaxDims(addVectors);
  occaKernelMaxOuterDims(addVectors);
  occaKernelMaxInnerDims(addVectors);
}

void testRun() {
  // TODO: Add test kernels
}
