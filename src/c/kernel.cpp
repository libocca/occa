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
    modeKernel.arguments.push_back(
      occa::c::kernelArg(arg)
    );
  }

  kernel_.run();
}

OCCA_END_EXTERN_C
