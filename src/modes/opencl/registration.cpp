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

#if OCCA_OPENCL_ENABLED

#include "occa/modes/opencl/registration.hpp"

namespace occa {
  namespace opencl {
    modeInfo::modeInfo() {}

    void modeInfo::init() {}

    styling::section& modeInfo::getDescription() {
      static styling::section section("OpenCL");
      if (section.size() == 0) {
        int platformCount = getPlatformCount();
        for (int pID = 0; pID < platformCount; ++pID) {
          int deviceCount = getDeviceCountInPlatform(pID);
          for (int dID = 0; dID < deviceCount; ++dID) {
            udim_t bytes         = getDeviceMemorySize(pID, dID);
            std::string bytesStr = stringifyBytes(bytes);

            section
              .add("Device Name"  , deviceName(pID, dID))
              .add("Driver Vendor", info::vendor(deviceVendor(pID,dID)))
              .add("Platform ID"  , toString(pID))
              .add("Device ID"    , toString(dID))
              .add("Memory"       , bytesStr)
              .addDivider();
          }
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

    occa::mode<opencl::modeInfo,
               opencl::device,
               opencl::kernel,
               opencl::memory> mode("OpenCL");
  }
}

#endif
