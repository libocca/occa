#define OCCA_DISABLE_VARIADIC_MACROS

#include <occa.h>
#include <occa.hpp>

#include <occa/internal/io.hpp>
#include <occa/internal/c/types.hpp>
#include <occa/internal/utils/testing.hpp>

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

  occaFree(&addVectors);

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
  occaJson props = occaKernelGetProperties(addVectors);
  occaType mode = occaJsonObjectGet(props, "mode", occaUndefined);
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

  const char *hash = occaKernelHash(addVectors);
  ASSERT_TRUE(hash != NULL);
  ::free((void*) hash);

  hash = occaKernelFullHash(addVectors);
  ASSERT_TRUE(hash != NULL);
  ::free((void*) hash);

  occaKernelMaxDims(addVectors);
  occaKernelMaxOuterDims(addVectors);
  occaKernelMaxInnerDims(addVectors);
}

void testRun() {
  std::string argKernelFile = (
    occa::env::OCCA_DIR + "tests/files/argKernel.okl"
  );
  occaJson kernelProps = occaJsonParse(
    "{type_validation: false}"
  );
  occaKernel argKernel = (
    occaBuildKernel(argKernelFile.c_str(),
                    "argKernel",
                    kernelProps)
  );
  occaFree(&kernelProps);

  // Dummy dims
  occaDim outerDims, innerDims;
  outerDims.x = innerDims.x = 1;
  outerDims.y = innerDims.y = 1;
  outerDims.z = innerDims.z = 1;

  occaKernelSetRunDims(argKernel,
                       outerDims,
                       innerDims);

  int value = 1;
  occaMemory mem = occaMalloc(1 * sizeof(int), &value, occaDefault);
  value = 2;

  int xy[2] = {13, 14};
  std::string str = "fifteen";

  // Good argument types
  occaKernelRunN(
    argKernel, 14,
    occaNull,
    mem,
    occaInt8(3),
    occaUInt8(4),
    occaInt16(5),
    occaUInt16(6),
    occaInt32(7),
    occaUInt32(8),
    occaInt64(9),
    occaUInt64(10),
    occaFloat(11.0),
    occaDouble(12.0),
    occaStruct(xy, sizeof(xy)),
    occaString(str.c_str())
  );

  // Manual argument insertion
  occaKernelClearArgs(argKernel);
  occaKernelPushArg(argKernel, occaNull);
  occaKernelPushArg(argKernel, mem);
  occaKernelPushArg(argKernel, occaInt8(3));
  occaKernelPushArg(argKernel, occaUInt8(4));
  occaKernelPushArg(argKernel, occaInt16(5));
  occaKernelPushArg(argKernel, occaUInt16(6));
  occaKernelPushArg(argKernel, occaInt32(7));
  occaKernelPushArg(argKernel, occaUInt32(8));
  occaKernelPushArg(argKernel, occaInt64(9));
  occaKernelPushArg(argKernel, occaUInt64(10));
  occaKernelPushArg(argKernel, occaFloat(11.0));
  occaKernelPushArg(argKernel, occaDouble(12.0));
  occaKernelPushArg(argKernel, occaStruct(xy, sizeof(xy)));
  occaKernelPushArg(argKernel, occaString(str.c_str()));
  occaKernelRunFromArgs(argKernel);

  // Test array call
  occaType args[14] = {
    occaNull,
    mem,
    occaInt8(3),
    occaUInt8(4),
    occaInt16(5),
    occaUInt16(6),
    occaInt32(7),
    occaUInt32(8),
    occaInt64(9),
    occaUInt64(10),
    occaFloat(11.0),
    occaDouble(12.0),
    occaStruct(xy, sizeof(xy)),
    occaString(str.c_str())
  };

  occaKernelRunWithArgs(argKernel, 14, args);

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
}
