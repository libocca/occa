#include <occa.hpp>
#include <occa/tools/testing.hpp>

occa::kernel addVectors;
const std::string addVectorsFile = (
  occa::env::OCCA_DIR + "tests/files/addVectors.okl"
);

void testInit();
void testInfo();
void testParsingFailure();
void testCompilingFailure();
void testArgumentFailure();
void testRun();

int main(const int argc, const char **argv) {
  addVectors = occa::buildKernel(addVectorsFile,
                                 "addVectors");

  testInit();
  testInfo();
  testParsingFailure();
  testCompilingFailure();
  testArgumentFailure();
  testRun();

  return 0;
}

void testInit() {
  occa::kernel addVectors2;
  ASSERT_FALSE(addVectors2.isInitialized());

  addVectors2 = addVectors;
  ASSERT_TRUE(addVectors2.isInitialized());
}

void testInfo() {
  occa::kernel addVectors2;

  ASSERT_EQ(addVectors2.mode(),
            "No Mode");

  addVectors2 = addVectors;

  ASSERT_EQ(addVectors.mode(),
            "Serial");

  const occa::properties &props = addVectors.properties();
  ASSERT_EQ(props["mode"].string(),
            "Serial");

  ASSERT_EQ(addVectors2.getDevice(),
            occa::host());

  ASSERT_EQ(addVectors2.name(),
            "addVectors");

  ASSERT_EQ(addVectors2.sourceFilename(),
            addVectorsFile.c_str());

  ASSERT_TRUE(
    occa::startsWith(addVectors2.binaryFilename(),
                     occa::io::cachePath())
  );

  addVectors2.maxDims();
  addVectors2.maxOuterDims();
  addVectors2.maxInnerDims();
}

void testParsingFailure() {
  occa::kernel badKernel;
  std::string badSource = (
    "@kernel foo {}"
  );

  // Bad C/C++ code
  ASSERT_THROW(
    badKernel = occa::buildKernelFromString(badSource,
                                            "foo");
  );

  badKernel = occa::buildKernelFromString(badSource,
                                          "foo",
                                          "silent: true");
  ASSERT_FALSE(badKernel.isInitialized());

  // Incorrect OKL
  badSource = (
    "@kernel void foo(int i) {}"
  );

  ASSERT_THROW(
    badKernel = occa::buildKernelFromString(badSource,
                                            "foo");
  );

  badKernel = occa::buildKernelFromString(badSource,
                                          "foo",
                                          "silent: true");
  ASSERT_FALSE(badKernel.isInitialized());
}

void testCompilingFailure() {
  // Good code, bad syntax (undefined N)
  occa::kernel badKernel;
  std::string badSource = (
    "@kernel void foo() {"
    "  for (int i = 0; i < N; ++i; @tile(16, @outer, @inner)) {}"
    "}"
  );

  ASSERT_THROW(
    badKernel = occa::buildKernelFromString(badSource,
                                            "foo");
  );
}

void testArgumentFailure() {
  occa::kernel kernel = occa::buildKernelFromString(
    "@kernel void foo(int N, float *arg) {"
    "  for (int i = 0; i < N; ++i; @tile(16, @outer, @inner)) {}"
    "}",
    "foo",
    "type_validation: false"
  );

  const int N = 10;

  // Use wrong device
  occa::device dev("mode: 'Serial'");
  occa::memory arg = dev.malloc(N * sizeof(float));

  ASSERT_THROW(
    kernel(N, arg);
  );
}

void testRun() {
  std::string argKernelFile = (
    occa::env::OCCA_DIR + "tests/files/argKernel.okl"
  );
  occa::kernel argKernel = occa::buildKernel(argKernelFile,
                                             "argKernel",
                                             "type_validation: false");

  argKernel.setRunDims(occa::dim(1, 1, 1),
                       occa::dim(1, 1, 1));

  int value = 0;
  occa::memory mem = occa::malloc(1 * sizeof(int), &value);

  value = 1;
  int *uvaPtr = (int*) occa::umalloc(1 * sizeof(int), &value);

  int xy[2] = {12, 13};
  std::string str = "fourteen";

  argKernel(
    mem,
    uvaPtr,
    (int8_t) 2,
    (uint8_t) 3,
    (int16_t) 4,
    (uint16_t) 5,
    (int32_t) 6,
    (uint32_t) 7,
    (int64_t) 8,
    (uint64_t) 9,
    (float) 10.0,
    (double) 11.0,
    xy,
    str.c_str()
  );

  occa::freeUvaPtr(uvaPtr);
}
