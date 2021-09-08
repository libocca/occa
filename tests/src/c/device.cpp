#define OCCA_DISABLE_VARIADIC_MACROS

#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <occa.hpp>
#include <occa.h>

#include <occa/internal/c/types.hpp>
#include <occa/internal/io.hpp>
#include <occa/internal/utils/testing.hpp>

const std::string deviceStr = (
  "{"
  "  mode: 'Serial',"
  "  dkey: 1,"
  "  kernel: {"
  "    kkey: 2,"
  "  },"
  "  memory: {"
  "    mkey: 3,"
  "  },"
  "}"
);

occaJson props = occaJsonParse(
  deviceStr.c_str()
);

void testInit();
void testProperties();
void testMemoryMethods();
void testKernelMethods();
void testStreamMethods();
void testWrapMemory();

int main(const int argc, const char **argv) {
  srand(time(NULL));

  testInit();
  testProperties();
  testMemoryMethods();
  testKernelMethods();
  testStreamMethods();
  testWrapMemory();

  occaFree(&props);

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

  occaPrintTypeInfo(device);

  occaFree(&device);

  device = occaCreateDevice(
    occaString(deviceStr.c_str())
  );

  ASSERT_THROW(
    occaCreateDevice(occaNull);
  );

  ASSERT_TRUE(occaDeviceIsInitialized(device));

  occaDeviceFinish(device);

  occaFree(&device);
}
void testProperties() {
  occaDevice device = occaUndefined;

  ASSERT_EQ((const char*) occaDeviceMode(device),
            (const char*) "No Mode");

  device = occaCreateDevice(props);

  ASSERT_EQ((const char*) occaDeviceMode(device),
            (const char*) "Serial");

  occaJson deviceProps = occaDeviceGetProperties(device);
  ASSERT_TRUE(occaJsonObjectHas(deviceProps, "dkey"));

  occaJson kernelProps = occaDeviceGetKernelProperties(device);
  ASSERT_TRUE(occaJsonObjectHas(kernelProps, "kkey"));

  occaJson memoryProps = occaDeviceGetMemoryProperties(device);
  ASSERT_TRUE(occaJsonObjectHas(memoryProps, "mkey"));

  occaFree(&device);
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

  // Test malloc
  occaMemory mem1 = occaDeviceMalloc(device, memBytes, NULL, occaDefault);
  allocatedBytes += memBytes;

  occaPrintTypeInfo(mem1);

  ASSERT_EQ((size_t) occaDeviceMemoryAllocated(device),
            allocatedBytes);

  occaMemory mem2 = occaDeviceMalloc(device, memBytes, NULL, props);
  allocatedBytes += memBytes;

  ASSERT_EQ((size_t) occaDeviceMemoryAllocated(device),
            allocatedBytes);

  ASSERT_EQ((size_t) occaDeviceMemoryAllocated(device),
            allocatedBytes);

  ASSERT_EQ((size_t) occaDeviceMemoryAllocated(device),
            allocatedBytes);

  // Free
  occaFree(&mem1);
  allocatedBytes -= memBytes;

  ASSERT_EQ((size_t) occaDeviceMemoryAllocated(device),
            allocatedBytes);

  occaFree(&mem2);
  allocatedBytes -= memBytes;

  ASSERT_EQ((size_t) occaDeviceMemoryAllocated(device),
            allocatedBytes);

  ASSERT_EQ((size_t) occaDeviceMemoryAllocated(device),
            allocatedBytes);

  ASSERT_EQ((size_t) occaDeviceMemoryAllocated(device),
            allocatedBytes);

  occaFree(&device);
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

  occaPrintTypeInfo(addVectors);

  occaFree(&addVectors);

  addVectors = occaDeviceBuildKernel(device,
                                     addVectorsFile.c_str(),
                                     "addVectors",
                                     props);
  occaFree(&addVectors);

  // occaBuildFromString
  addVectors = occaDeviceBuildKernelFromString(device,
                                               addVectorsSource.c_str(),
                                               "addVectors",
                                               occaDefault);
  occaFree(&addVectors);

  addVectors = occaDeviceBuildKernelFromString(device,
                                               addVectorsSource.c_str(),
                                               "addVectors",
                                               props);
  occaFree(&addVectors);

  // occaBuildFromBinary
  addVectors = occaDeviceBuildKernelFromBinary(device,
                                               addVectorsBinaryFile.c_str(),
                                               "addVectors",
                                               occaDefault);
  occaFree(&addVectors);

  addVectors = occaDeviceBuildKernelFromBinary(device,
                                               addVectorsBinaryFile.c_str(),
                                               "addVectors",
                                               props);
  occaFree(&addVectors);

  occaFree(&device);
}

void testStreamMethods() {
  occaDevice device = occaCreateDevice(props);
  occaSetDevice(device);

  occaStream cStream = occaDeviceCreateStream(device, occaDefault);
  occa::stream stream = occa::c::stream(cStream);

  occaPrintTypeInfo(cStream);

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

  occaFree(&cStream);
  occaFree(&device);
}

void testWrapMemory() {
  occaDevice device = occaCreateDevice(props);
  const int entries = 10;
  const size_t bytes = entries * sizeof(int);

  occaMemory mem1 = occaMalloc(bytes, NULL, occaDefault);
  occaMemory mem2 = occaMemoryClone(mem1);

  ASSERT_EQ(occaMemorySize(mem1),
            occaMemorySize(mem2));

  ASSERT_NEQ(occa::c::memory(mem1),
             occa::c::memory(mem2));

  int *ptr = (int*) occaMemoryPtr(mem2);
  occaMemoryDetach(mem2);

  for (int i = 0; i < entries; ++i) {
    ptr[i] = i;
  }

  mem2 = occaDeviceWrapMemory(occaHost(),
                              ptr,
                              bytes,
                              occaDefault);

  mem2 = occaDeviceTypedWrapMemory(device,
                                   ptr,
                                   entries,
                                   occaDtypeInt,
                                   occaDefault);

  occaJson memProps = (
    occaJsonParse("{foo: 'bar'}")
  );

  mem2 = occaDeviceWrapMemory(occaHost(),
                              ptr,
                              bytes,
                              memProps);

  mem2 = occaDeviceTypedWrapMemory(device,
                                   ptr,
                                   entries,
                                   occaDtypeInt,
                                   memProps);

  occaFree(&device);
  occaFree(&memProps);
}
