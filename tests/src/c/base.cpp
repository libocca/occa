#define OCCA_DISABLE_VARIADIC_MACROS

#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <occa.hpp>
#include <occa.h>

#include <occa/internal/c/types.hpp>
#include <occa/internal/io.hpp>
#include <occa/internal/utils/testing.hpp>

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
  ASSERT_EQ(&occa::c::json(occaSettings()),
            &occa::settings());
}

void testDeviceMethods() {
  ASSERT_EQ(occa::c::device(occaHost()),
            occa::host());

  ASSERT_EQ(occa::c::device(occaGetDevice()),
            occa::getDevice());

  occa::device fakeDevice({
    {"mode", "Serial"},
    {"key", "value"}
  });
  occaSetDevice(occa::c::newOccaType(fakeDevice));
  ASSERT_EQ(occa::getDevice(),
            fakeDevice);

  occaSetDeviceFromString(
    "{"
    "  mode: 'Serial',"
    "  key: 'value',"
    "}"
  );
  occa::json &fakeProps = occa::c::json(occaDeviceProperties());
  ASSERT_EQ((std::string) fakeProps["key"],
            "value");

  occaFinish();
}

void testMemoryMethods() {
  size_t bytes = 10 * sizeof(int);
  occaJson props = (
    occaJsonParse("{a: 1, b: 2}")
  );

  // malloc
  occaMemory mem = occaMalloc(bytes,
                              NULL,
                              occaDefault);
  ASSERT_EQ(occaMemorySize(mem),
            bytes);
  occaFree(&mem);

  mem = occaMalloc(bytes,
                   NULL,
                   props);
  ASSERT_EQ(occaMemorySize(mem),
            bytes);
  occaFree(&mem);

  // umalloc
  void *ptr = occaUMalloc(bytes,
                          NULL,
                          occaDefault);
  occaFreeUvaPtr(ptr);

  ptr = occaUMalloc(bytes,
                    NULL,
                    props);
  occaFreeUvaPtr(ptr);

  occaFree(&props);
}

void testKernelMethods() {
  const std::string addVectorsFile = (
    occa::env::OCCA_DIR + "tests/files/addVectors.okl"
  );
  const std::string addVectorsSource = (
    occa::io::read(addVectorsFile)
  );

  occaJson props = occaCreateJson();
  occaJsonObjectSet(props, "defines/foo", occaInt(3));

  // occaBuildKernel
  occaKernel addVectors = occaBuildKernel(addVectorsFile.c_str(),
                                          "addVectors",
                                          occaDefault);

  const std::string addVectorsBinaryFile = (
    occa::c::kernel(addVectors).binaryFilename()
  );

  occaFree(&addVectors);

  addVectors = occaBuildKernel(addVectorsFile.c_str(),
                               "addVectors",
                               props);
  occaFree(&addVectors);

  // occaBuildFromString
  addVectors = occaBuildKernelFromString(addVectorsSource.c_str(),
                                         "addVectors",
                                         occaDefault);
  occaFree(&addVectors);

  addVectors = occaBuildKernelFromString(addVectorsSource.c_str(),
                                         "addVectors",
                                         props);
  occaFree(&addVectors);

  // occaBuildFromBinary
  addVectors = occaBuildKernelFromBinary(addVectorsBinaryFile.c_str(),
                                         "addVectors",
                                         occaDefault);
  occaFree(&addVectors);

  addVectors = occaBuildKernelFromBinary(addVectorsBinaryFile.c_str(),
                                         "addVectors",
                                         props);
  occaFree(&addVectors);

  occaFree(&props);
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

  occaFree(&cStream);
}
