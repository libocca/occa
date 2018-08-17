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

#include <occa/defines.hpp>

#if OCCA_HIP_ENABLED

#include <occa/mode/hip/registration.hpp>
#include <occa/mode/hip/utils.hpp>

namespace occa {
  namespace hip {
    modeInfo::modeInfo() {}

    void modeInfo::init() {
      hip::init();
    }

    styling::section& modeInfo::getDescription() {
      static styling::section section("HIP");
      if (section.size() == 0) {
        hipDevice_t device;
        char deviceName[256];
        int deviceCount = hip::getDeviceCount();
        for (int i = 0; i < deviceCount; ++i) {
          hipDeviceProp_t props;
          OCCA_HIP_ERROR("Getting device properties",
                         hipGetDeviceProperties(&props, i));
          strcpy(deviceName, props.name);

          const udim_t bytes         = props.totalGlobalMem;
          const std::string bytesStr = stringifyBytes(bytes);

          section
            .add("Device ID"  ,  toString(i))
            .add("Arch"       , "gfx" + toString(props.gcnArch))
            .add("Device Name",  deviceName)
            .add("Memory"     ,  bytesStr)
            .addDivider();
        }
        // Remove last divider
        section.groups.pop_back();
      }
      return section;
    }

    occa::mode<hip::modeInfo,
               hip::device> mode("HIP");
  }
}

#endif
