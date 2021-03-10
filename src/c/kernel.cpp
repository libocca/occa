#include <cstring>
#include <stdarg.h>

#include <occa/internal/c/types.hpp>
#include <occa/c/kernel.h>
#include <occa/internal/core/kernel.hpp>

OCCA_START_EXTERN_C

bool occaKernelIsInitialized(occaKernel kernel) {
  return (int) occa::c::kernel(kernel).isInitialized();
}

occaJson occaKernelGetProperties(occaKernel kernel) {
  return occa::c::newOccaType(
    occa::c::kernel(kernel).properties(),
    false
  );
}

occaDevice occaKernelGetDevice(occaKernel kernel) {
  return occa::c::newOccaType(
    occa::c::kernel(kernel).getDevice()
  );
}

const char* occaKernelName(occaKernel kernel) {
  return occa::c::kernel(kernel).name().c_str();
}

const char* occaKernelSourceFilename(occaKernel kernel) {
  return occa::c::kernel(kernel).sourceFilename().c_str();
}

const char* occaKernelBinaryFilename(occaKernel kernel) {
  return occa::c::kernel(kernel).binaryFilename().c_str();
}

const char* occaKernelHash(occaKernel kernel) {
  occa::hash_t hash = occa::c::kernel(kernel).hash();
  if (!hash.isInitialized()) {
    return (const char*) NULL;
  }

  std::string hashStr = hash.getString();

  const int charCount = (int) hashStr.size();
  char *c_str = (char*) ::malloc(charCount);
  ::memcpy(c_str, hashStr.c_str(), charCount);

  return c_str;
}

const char* occaKernelFullHash(occaKernel kernel) {
  occa::hash_t hash = occa::c::kernel(kernel).hash();
  if (!hash.isInitialized()) {
    return (const char*) NULL;
  }

  std::string hashStr = hash.getFullString();

  const int charCount = (int) hashStr.size();
  char *c_str = (char*) ::malloc(charCount);
  ::memcpy(c_str, hashStr.c_str(), charCount);

  return c_str;
}

int occaKernelMaxDims(occaKernel kernel) {
  return occa::c::kernel(kernel).maxDims();
}

occaDim occaKernelMaxOuterDims(occaKernel kernel) {
  occa::dim dims = occa::c::kernel(kernel).maxOuterDims();
  occaDim cDims;
  cDims.x = dims.x;
  cDims.y = dims.y;
  cDims.z = dims.z;
  return cDims;
}

occaDim occaKernelMaxInnerDims(occaKernel kernel) {
  occa::dim dims = occa::c::kernel(kernel).maxInnerDims();
  occaDim cDims;
  cDims.x = dims.x;
  cDims.y = dims.y;
  cDims.z = dims.z;
  return cDims;
}

void occaKernelSetRunDims(occaKernel kernel,
                          occaDim outerDims,
                          occaDim innerDims) {
  occa::c::kernel(kernel).setRunDims(
    occa::dim(outerDims.x, outerDims.y, outerDims.z),
    occa::dim(innerDims.x, innerDims.y, innerDims.z)
  );
}

void occaKernelPushArg(occaKernel kernel,
                       occaType arg) {
  if (&arg != &occaNull) {
    occa::c::kernel(kernel).pushArg(
      occa::c::kernelArg(arg)
    );
  } else {
    occa::c::kernel(kernel).pushArg(occa::null);
  }
}

void occaKernelClearArgs(occaKernel kernel) {
  occa::c::kernel(kernel).clearArgs();
}

void occaKernelRunFromArgs(occaKernel kernel) {
  occa::c::kernel(kernel).run();
}

// `occaKernelRun` is reserved for a variadic macro
//    which is more user-friendly
void occaKernelRunN(occaKernel kernel,
                    const int argc,
                    ...) {
  va_list args;
  va_start(args, argc);
  occaKernelVaRun(kernel, argc, args);
  va_end(args);
}

void occaKernelVaRun(occaKernel kernel,
                     const int argc,
                     va_list args) {
  occa::kernel kernel_ = occa::c::kernel(kernel);
  OCCA_ERROR("Uninitialized kernel",
             kernel_.isInitialized());

  occa::modeKernel_t &modeKernel = *(kernel_.getModeKernel());
  modeKernel.arguments.clear();
  modeKernel.arguments.reserve(argc);

  va_list runArgs;
  va_copy(runArgs, args);
  for (int i = 0; i < argc; ++i) {
    occaType arg = va_arg(runArgs, occaType);
    modeKernel.pushArgument(
      occa::c::kernelArg(arg)
    );
  }
  va_end(runArgs);

  kernel_.run();
}

void occaKernelRunWithArgs(occaKernel kernel,
                           const int argc,
                           occaType *args) {
  occa::kernel kernel_ = occa::c::kernel(kernel);
  OCCA_ERROR("Uninitialized kernel",
             kernel_.isInitialized());

  occa::modeKernel_t &modeKernel = *(kernel_.getModeKernel());
  modeKernel.arguments.clear();
  modeKernel.arguments.reserve(argc);

  for (int i = 0; i < argc; ++i) {
    modeKernel.pushArgument(
      occa::c::kernelArg(args[i])
    );
  }

  kernel_.run();
}

OCCA_END_EXTERN_C
