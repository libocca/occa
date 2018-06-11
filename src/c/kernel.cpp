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
    occa::c::kernel(kernel).properties()
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
                                     occaDim groups,
                                     occaDim items) {

  occa::dim groupDim(groups.x, groups.y, groups.z);
  occa::dim itemDim(items.x, items.y, items.z);

  occa::c::kernel(kernel).setRunDims(groupDim, itemDim);
}

// `occaKernelRun` is reserved for a variadic macro
//    which is more user-friendly
void OCCA_RFUNC occaKernelRunN(occaKernel kernel,
                               const int argc,
                               ...) {
  occa::kernel kernel_ = occa::c::kernel(kernel);
  OCCA_ERROR("Uninitialized kernel",
             kernel_.isInitialized());

  occa::kernel_v &kHandle = *(kernel_.getKHandle());
  kHandle.arguments.clear();
  kHandle.arguments.reserve(argc);

  va_list args;
  va_start(args, argc);
  for (int i = 0; i < argc; ++i) {
    occa::kernelArg kArg;

    occaType arg = va_arg(args, occaType);
    OCCA_ERROR("A non-occaType argument was passed",
               arg.magicHeader == OCCA_C_TYPE_MAGIC_HEADER);

    switch (arg.type) {
    case occa::c::typeType::none: {
      kArg.add(NULL, false, false); break;
    }
    case occa::c::typeType::ptr: {
      kArg.add(arg.value.ptr,
               arg.bytes,
               false, false);
      break;
    }
    case occa::c::typeType::int8_: {
      kArg = occa::kernelArg(arg.value.int8_);
      break;
    }
    case occa::c::typeType::uint8_: {
      kArg = occa::kernelArg(arg.value.uint8_);
      break;
    }
    case occa::c::typeType::int16_: {
      kArg = occa::kernelArg(arg.value.int16_);
      break;
    }
    case occa::c::typeType::uint16_: {
      kArg = occa::kernelArg(arg.value.uint16_);
      break;
    }
    case occa::c::typeType::int32_: {
      kArg = occa::kernelArg(arg.value.int32_);
      break;
    }
    case occa::c::typeType::uint32_: {
      kArg = occa::kernelArg(arg.value.uint32_);
      break;
    }
    case occa::c::typeType::int64_: {
      kArg = occa::kernelArg(arg.value.int64_);
      break;
    }
    case occa::c::typeType::uint64_: {
      kArg = occa::kernelArg(arg.value.uint64_);
      break;
    }
    case occa::c::typeType::float_: {
      kArg = occa::kernelArg(arg.value.float_);
      break;
    }
    case occa::c::typeType::double_: {
      kArg = occa::kernelArg(arg.value.double_);
      break;
    }
    case occa::c::typeType::struct_: {
      kArg.add(arg.value.ptr,
               arg.bytes,
               false, false);
      break;
    }
    case occa::c::typeType::string: {
      kArg.add(arg.value.ptr,
               arg.bytes,
               false, false);
      break;
    }
    case occa::c::typeType::memory:
      kArg = occa::kernelArg(occa::c::memory(arg));
      break;
    case occa::c::typeType::device:
      OCCA_FORCE_ERROR("Unable to pass an occaDevice as a kernel argument"); break;
    case occa::c::typeType::kernel:
      OCCA_FORCE_ERROR("Unable to pass an occaKernel as a kernel argument"); break;
    case occa::c::typeType::properties:
      OCCA_FORCE_ERROR("Unable to pass an occaProperties as a kernel argument"); break;
    }

    kHandle.arguments.push_back(kArg);
  }

  kernel_.run();
}

OCCA_END_EXTERN_C
