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
  ASSERT_EQ((const char*) occaJsonGetString(mode),
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
  std::string argKernelFile = (
    occa::env::OCCA_DIR + "tests/files/argKernel.okl"
  );
  occaKernel argKernel = (
    occaBuildKernel(argKernelFile.c_str(),
                    "argKernel",
                    occaDefault)
  );

  // Dummy dims
  occaDim outerDims, innerDims;
  outerDims.x = innerDims.x = 1;
  outerDims.y = innerDims.y = 1;
  outerDims.z = innerDims.z = 1;

  occaKernelSetRunDims(argKernel,
                       outerDims,
                       innerDims);

  int value = 0;
  occaMemory mem = occaMalloc(1 * sizeof(int), &value, occaDefault);
  value = 1;
  int *uvaPtr = (int*) occaUMalloc(1 * sizeof(int), &value, occaDefault);

  int xy[2] = {12, 13};
  std::string str = "fourteen";

  // Good argument types
  occaKernelRunN(
    argKernel, 14,
    mem,
    occaPtr(uvaPtr),
    occaInt8(2),
    occaUInt8(3),
    occaInt16(4),
    occaUInt16(5),
    occaInt32(6),
    occaUInt32(7),
    occaInt64(8),
    occaUInt64(9),
    occaFloat(10.0),
    occaDouble(11.0),
    occaStruct(xy, sizeof(xy)),
    occaString(str.c_str())
  );

  // Manual argument insertion
  occaKernelClearArgs(argKernel);
  occaKernelPushArg(argKernel, mem);
  occaKernelPushArg(argKernel, occaPtr(uvaPtr));
  occaKernelPushArg(argKernel, occaInt8(2));
  occaKernelPushArg(argKernel, occaUInt8(3));
  occaKernelPushArg(argKernel, occaInt16(4));
  occaKernelPushArg(argKernel, occaUInt16(5));
  occaKernelPushArg(argKernel, occaInt32(6));
  occaKernelPushArg(argKernel, occaUInt32(7));
  occaKernelPushArg(argKernel, occaInt64(8));
  occaKernelPushArg(argKernel, occaUInt64(9));
  occaKernelPushArg(argKernel, occaFloat(10.0));
  occaKernelPushArg(argKernel, occaDouble(11.0));
  occaKernelPushArg(argKernel, occaStruct(xy, sizeof(xy)));
  occaKernelPushArg(argKernel, occaString(str.c_str()));
  occaKernelRunFromArgs(argKernel);

  // Bad argument types
  ASSERT_THROW(
    occaKernelRunN(argKernel, 1, occaGetDevice());
  );
  ASSERT_THROW(
    occaKernelRunN(argKernel, 1, argKernel);
  );
  ASSERT_THROW(
    occaKernelRunN(argKernel, 1, occaSettings());
  );
  ASSERT_THROW(
    occaKernelRunN(argKernel, 1, occaUndefined);
  );
  ASSERT_THROW(
    occaKernelRunN(argKernel, 1, occaDefault);
  );
  ASSERT_THROW(
    occaKernelRunN(argKernel, 1, uvaPtr);
  );
}
