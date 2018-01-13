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

#include "occa/defines.hpp"

#if OCCA_CUDA_ENABLED

#include "occa/modes/cuda/registration.hpp"
#include "occa/modes/cuda/utils.hpp"

namespace occa {
  namespace cuda {
    modeInfo::modeInfo() {}

    void modeInfo::init() {
      cuda::init();
    }

    styling::section& modeInfo::getDescription() {
      static styling::section section("CUDA");
      if (section.size() == 0) {
        char deviceName[1024];
        int deviceCount = cuda::getDeviceCount();
        for (int i = 0; i < deviceCount; ++i) {
          const udim_t bytes         = getDeviceMemorySize(getDevice(i));
          const std::string bytesStr = stringifyBytes(bytes);

          OCCA_CUDA_ERROR("Getting Device Name",
                          cuDeviceGetName(deviceName, 1024, i));

          section
            .add("Device ID"  ,  toString(i))
            .add("Device Name",  deviceName)
            .add("Memory"     ,  bytesStr)
            .addDivider();
        }
        // Remove last divider
        section.groups.pop_back();
      }
      return section;
    }

    occa::properties& modeInfo::getProperties() {
      static occa::properties properties;
      return properties;
    }

    occa::mode<cuda::modeInfo,
               cuda::device,
               cuda::kernel,
               cuda::memory> mode("CUDA");
  }
}

#endif
