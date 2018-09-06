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

const std::string deviceStr = (
  "mode: 'Serial',"
  "dkey: 1,"
  "kernel: {"
  "  kkey: 2,"
  "},"
  "memory: {"
  "  mkey: 3,"
  "},"
);

occaProperties props = occaCreatePropertiesFromString(
  deviceStr.c_str()
);

void testInit();
void testProperties();
void testMemoryMethods();
void testKernelMethods();
void testStreamMethods();

int main(const int argc, const char **argv) {
  srand(time(NULL));

  testInit();
  testProperties();
  testMemoryMethods();
  testKernelMethods();
  testStreamMethods();

  occaFree(props);

  return 0;
}

void testInit() {
  occaDevice device = occaUndefined;
  ASSERT_TRUE(occaIsUndefined(device));
  ASSERT_EQ(device.type,
            OCCA_UNDEFINED);

  device = occaCreateDevice(props);
  ASSERT_FALSE(occaIsUndefined(device));
  ASSERT_EQ(device.type,
            OCCA_DEVICE);

  occaFree(device);

  device = occaCreateDevice(
    occaString(deviceStr.c_str())
  );

  ASSERT_THROW(
    occaCreateDevice(occaNull);
  );

  ASSERT_TRUE(occaDeviceIsInitialized(device));

  occaDeviceFinish(device);

  occaFree(device);
}
void testProperties() {
  occaDevice device = occaUndefined;

  ASSERT_EQ((const char*) occaDeviceMode(device),
            (const char*) "No Mode");

  device = occaCreateDevice(props);

  ASSERT_EQ((const char*) occaDeviceMode(device),
            (const char*) "Serial");

  occaProperties deviceProps = occaDeviceGetProperties(device);
  ASSERT_TRUE(occaPropertiesHas(deviceProps, "dkey"));

  occaProperties kernelProps = occaDeviceGetKernelProperties(device);
  ASSERT_TRUE(occaPropertiesHas(kernelProps, "kkey"));

  occaProperties memoryProps = occaDeviceGetMemoryProperties(device);
  ASSERT_TRUE(occaPropertiesHas(memoryProps, "mkey"));

  occaFree(device);
}

void testMemoryMethods() {
  occaDevice device = occaCreateDevice(props);

  // Info
  occaDeviceMemorySize(device);

  ASSERT_FALSE(occaDeviceHasSeparateMemorySpace(device));

  size_t allocatedBytes = 0;
  size_t memBytes = 10 * sizeof(int);

  ASSERT_EQ((size_t) occaDeviceMemoryAllocated(device),
            allocatedBytes);

  // Test malloc + umalloc
  occaMemory mem1 = occaDeviceMalloc(device, memBytes, NULL, occaDefault);
  allocatedBytes += memBytes;

  ASSERT_EQ((size_t) occaDeviceMemoryAllocated(device),
            allocatedBytes);

  occaMemory mem2 = occaDeviceMalloc(device, memBytes, NULL, props);
  allocatedBytes += memBytes;

  ASSERT_EQ((size_t) occaDeviceMemoryAllocated(device),
            allocatedBytes);

  void *ptr1 = occaDeviceUMalloc(device, memBytes, NULL, occaDefault);
  allocatedBytes += memBytes;

  ASSERT_EQ((size_t) occaDeviceMemoryAllocated(device),
            allocatedBytes);

  void *ptr2 = occaDeviceUMalloc(device, memBytes, NULL, props);
  allocatedBytes += memBytes;

  ASSERT_EQ((size_t) occaDeviceMemoryAllocated(device),
            allocatedBytes);

  // Free
  occaFree(mem1);
  allocatedBytes -= memBytes;

  ASSERT_EQ((size_t) occaDeviceMemoryAllocated(device),
            allocatedBytes);

  occaFree(mem2);
  allocatedBytes -= memBytes;

  ASSERT_EQ((size_t) occaDeviceMemoryAllocated(device),
            allocatedBytes);

  occaFreeUvaPtr(ptr1);
  allocatedBytes -= memBytes;

  ASSERT_EQ((size_t) occaDeviceMemoryAllocated(device),
            allocatedBytes);


  occaFreeUvaPtr(ptr2);
  allocatedBytes -= memBytes;

  ASSERT_EQ((size_t) occaDeviceMemoryAllocated(device),
            allocatedBytes);

  occaFree(device);
}

void testKernelMethods() {
  occaDevice device = occaCreateDevice(props);

  const std::string addVectorsFile = (
    occa::env::OCCA_DIR + "tests/files/addVectors.okl"
  );
  const std::string addVectorsSource = (
    occa::io::read(addVectorsFile)
  );

  // occaBuildKernel
  occaKernel addVectors = occaDeviceBuildKernel(device,
                                                addVectorsFile.c_str(),
                                                "addVectors",
                                                occaDefault);

  const std::string addVectorsBinaryFile = (
    occa::c::kernel(addVectors).binaryFilename()
  );

  occaFree(addVectors);

  addVectors = occaDeviceBuildKernel(device,
                                     addVectorsFile.c_str(),
                                     "addVectors",
                                     props);
  occaFree(addVectors);

  // occaBuildFromString
  addVectors = occaDeviceBuildKernelFromString(device,
                                               addVectorsSource.c_str(),
                                               "addVectors",
                                               occaDefault);
  occaFree(addVectors);

  addVectors = occaDeviceBuildKernelFromString(device,
                                               addVectorsSource.c_str(),
                                               "addVectors",
                                               props);
  occaFree(addVectors);

  // occaBuildFromBinary
  addVectors = occaDeviceBuildKernelFromBinary(device,
                                               addVectorsBinaryFile.c_str(),
                                               "addVectors",
                                               occaDefault);
  occaFree(addVectors);

  addVectors = occaDeviceBuildKernelFromBinary(device,
                                               addVectorsBinaryFile.c_str(),
                                               "addVectors",
                                               props);
  occaFree(addVectors);

  occaFree(device);
}

void testStreamMethods() {
  occaDevice device = occaCreateDevice(props);
  occaSetDevice(device);

  occaStream cStream = occaDeviceCreateStream(device, occaDefault);
  occa::stream stream = occa::c::stream(cStream);

  occaDeviceSetStream(device, cStream);

  ASSERT_EQ(stream.getModeStream(),
            occa::getStream().getModeStream());

  ASSERT_EQ(stream.getModeStream(),
            occa::c::stream(occaDeviceGetStream(device)).getModeStream());

  // Start tagging
  double outerStart = occa::sys::currentTime();
  occaStreamTag startTag = occaDeviceTagStream(device);
  double innerStart = occa::sys::currentTime();

  // Wait 0.3 - 0.5 seconds
  ::usleep((3 + (rand() % 3)) * 100000);

  // End tagging
  double innerEnd = occa::sys::currentTime();
  occaStreamTag endTag = occaDeviceTagStream(device);
  occaDeviceWaitForTag(device, endTag);
  double outerEnd = occa::sys::currentTime();

  double tagTime = occaDeviceTimeBetweenTags(device, startTag, endTag);

  ASSERT_GE(outerEnd - outerStart,
            tagTime);
  ASSERT_LE(innerEnd - innerStart,
            tagTime);

  occaFree(cStream);
  occaFree(device);
}
