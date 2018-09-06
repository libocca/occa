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

#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <occa.hpp>
#include <occa.h>
#include <occa/c/types.hpp>
#include <occa/tools/testing.hpp>

void testGlobals();
void testDeviceMethods();
void testMemoryMethods();
void testKernelMethods();
void testStreamMethods();

int main(const int argc, const char **argv) {
  srand(time(NULL));

  testGlobals();
  testDeviceMethods();
  testMemoryMethods();
  testKernelMethods();
  testStreamMethods();

  return 0;
}

void testGlobals() {
  ASSERT_EQ(&occa::c::properties(occaSettings()),
            &occa::settings());
}

void testDeviceMethods() {
  ASSERT_EQ(occa::c::device(occaHost()),
            occa::host());

  ASSERT_EQ(occa::c::device(occaGetDevice()),
            occa::getDevice());

  occa::device fakeDevice("mode: 'Serial',"
                          "key: 'value'");
  occaSetDevice(occa::c::newOccaType(fakeDevice));
  ASSERT_EQ(occa::getDevice(),
            fakeDevice);

  occaSetDeviceFromString("mode: 'Serial',"
                          "key: 'value'");
  occa::properties &fakeProps = occa::c::properties(occaDeviceProperties());
  ASSERT_EQ((std::string) fakeProps["key"],
            "value");

  occaLoadKernels("lib");

  occaFinish();
}

void testMemoryMethods() {
  size_t bytes = 10 * sizeof(int);
  occaProperties props = (
    occaCreatePropertiesFromString("a: 1, b: 2")
  );

  // malloc
  occaMemory mem = occaMalloc(bytes,
                              NULL,
                              occaDefault);
  ASSERT_EQ(occaMemorySize(mem),
            bytes);
  occaFree(mem);

  mem = occaMalloc(bytes,
                   NULL,
                   props);
  ASSERT_EQ(occaMemorySize(mem),
            bytes);
  occaFree(mem);

  // umalloc
  void *ptr = occaUMalloc(bytes,
                          NULL,
                          occaDefault);
  occaFreeUvaPtr(ptr);

  ptr = occaUMalloc(bytes,
                    NULL,
                    props);
  occaFreeUvaPtr(ptr);

  occaFree(props);
}

void testKernelMethods() {
  const std::string addVectorsFile = (
    occa::env::OCCA_DIR + "tests/files/addVectors.okl"
  );
  const std::string addVectorsSource = (
    occa::io::read(addVectorsFile)
  );

  occaProperties props = occaCreateProperties();
  occaPropertiesSet(props, "defines/foo", occaInt(3));

  // occaBuildKernel
  occaKernel addVectors = occaBuildKernel(addVectorsFile.c_str(),
                                          "addVectors",
                                          occaDefault);

  const std::string addVectorsBinaryFile = (
    occa::c::kernel(addVectors).binaryFilename()
  );

  occaFree(addVectors);

  addVectors = occaBuildKernel(addVectorsFile.c_str(),
                               "addVectors",
                               props);
  occaFree(addVectors);

  // occaBuildFromString
  addVectors = occaBuildKernelFromString(addVectorsSource.c_str(),
                                         "addVectors",
                                         occaDefault);
  occaFree(addVectors);

  addVectors = occaBuildKernelFromString(addVectorsSource.c_str(),
                                         "addVectors",
                                         props);
  occaFree(addVectors);

  // occaBuildFromBinary
  addVectors = occaBuildKernelFromBinary(addVectorsBinaryFile.c_str(),
                                         "addVectors",
                                         occaDefault);
  occaFree(addVectors);

  addVectors = occaBuildKernelFromBinary(addVectorsBinaryFile.c_str(),
                                         "addVectors",
                                         props);
  occaFree(addVectors);

  occaFree(props);
}

void testStreamMethods() {
  occaStream cStream = occaCreateStream(occaDefault);
  occa::stream stream = occa::c::stream(cStream);

  occaSetStream(cStream);

  ASSERT_EQ(stream.getModeStream(),
            occa::getStream().getModeStream());

  ASSERT_EQ(stream.getModeStream(),
            occa::c::stream(occaGetStream()).getModeStream());

  // Start tagging
  double outerStart = occa::sys::currentTime();
  occaStreamTag startTag = occaTagStream();
  double innerStart = occa::sys::currentTime();

  // Wait 0.3 - 0.5 seconds
  ::usleep((3 + (rand() % 3)) * 100000);

  // End tagging
  double innerEnd = occa::sys::currentTime();
  occaStreamTag endTag = occaTagStream();
  occaWaitForTag(endTag);
  double outerEnd = occa::sys::currentTime();

  double tagTime = occaTimeBetweenTags(startTag, endTag);

  ASSERT_GE(outerEnd - outerStart,
            tagTime);
  ASSERT_LE(innerEnd - innerStart,
            tagTime);

  occaFree(cStream);
}
