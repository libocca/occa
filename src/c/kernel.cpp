#include <stdarg.h>

#include <occa/c/types.hpp>
#include <occa/c/kernel.h>

OCCA_START_EXTERN_C

int OCCA_RFUNC occaKernelIsInitialized(occaKernel kernel) {
  return (int) occa::c::kernel(kernel).isInitialized();
}

occaProperties OCCA_RFUNC occaKernelGetProperties(occaKernel kernel) {
  return occa::c::newOccaType(
    occa::c::kernel(kernel).properties(),
    false
  );
}

occaDevice OCCA_RFUNC occaKernelGetDevice(occaKernel kernel) {
  return occa::c::newOccaType(
    occa::c::kernel(kernel).getDevice()
  );
}

const char* OCCA_RFUNC occaKernelName(occaKernel kernel) {
  return occa::c::kernel(kernel).name().c_str();
}

const char* OCCA_RFUNC occaKernelSourceFilename(occaKernel kernel) {
  return occa::c::kernel(kernel).sourceFilename().c_str();
}

const char* OCCA_RFUNC occaKernelBinaryFilename(occaKernel kernel) {
  return occa::c::kernel(kernel).binaryFilename().c_str();
}

int OCCA_RFUNC occaKernelMaxDims(occaKernel kernel) {
  return occa::c::kernel(kernel).maxDims();
}

occaDim OCCA_RFUNC occaKernelMaxOuterDims(occaKernel kernel) {
  occa::dim dims = occa::c::kernel(kernel).maxOuterDims();
  occaDim cDims;
  cDims.x = dims.x;
  cDims.y = dims.y;
  cDims.z = dims.z;
  return cDims;
}

occaDim OCCA_RFUNC occaKernelMaxInnerDims(occaKernel kernel) {
  occa::dim dims = occa::c::kernel(kernel).maxInnerDims();
  occaDim cDims;
  cDims.x = dims.x;
  cDims.y = dims.y;
  cDims.z = dims.z;
  return cDims;
}

void OCCA_RFUNC occaKernelSetRunDims(occaKernel kernel,
                                     occaDim outerDims,
                                     occaDim innerDims) {
  occa::c::kernel(kernel).setRunDims(
    occa::dim(outerDims.x, outerDims.y, outerDims.z),
    occa::dim(innerDims.x, innerDims.y, innerDims.z)
  );
}

OCCA_LFUNC void OCCA_RFUNC occaKernelPushArg(occaKernel kernel,
                                             occaType arg) {
  occa::c::kernel(kernel).pushArg(
    occa::c::kernelArg(arg)
  );
}

OCCA_LFUNC void OCCA_RFUNC occaKernelClearArgs(occaKernel kernel) {
  occa::c::kernel(kernel).clearArgs();
}

OCCA_LFUNC void OCCA_RFUNC occaKernelRunFromArgs(occaKernel kernel) {
  occa::c::kernel(kernel).run();
}

// `occaKernelRun` is reserved for a variadic macro
//    which is more user-friendly
void OCCA_RFUNC occaKernelRunN(occaKernel kernel,
                               const int argc,
                               ...) {
  va_list args;
  va_start(args, argc);
  occaKernelVaRun(kernel, argc, args);
}

void OCCA_RFUNC occaKernelVaRun(occaKernel kernel,
                                const int argc,
                                va_list args) {
  occa::kernel kernel_ = occa::c::kernel(kernel);
  OCCA_ERROR("Uninitialized kernel",
             kernel_.isInitialized());

  occa::modeKernel_t &modeKernel = *(kernel_.getModeKernel());
  modeKernel.arguments.clear();
  modeKernel.arguments.reserve(argc);

  for (int i = 0; i < argc; ++i) {
    occaType arg = va_arg(args, occaType);
    modeKernel.pushArgument(
      occa::c::kernelArg(arg)
    );
  }

  kernel_.run();
}

OCCA_END_EXTERN_C
