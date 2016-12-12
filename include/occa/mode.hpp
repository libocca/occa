/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2016 David Medina and Tim Warburton
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

#ifndef OCCA_MODE_HEADER
#define OCCA_MODE_HEADER

#include <iostream>
#include <map>

#include "occa/defines.hpp"
#include "occa/tools/properties.hpp"

namespace occa {
  class kernel_v; class kernel;
  class memory_v; class memory;
  class device_v; class device;

  class mode_v;

  typedef std::map<std::string,mode_v*> strToModeMap_t;
  typedef strToModeMap_t::iterator      strToModeMapIterator;

  strToModeMap_t& modeMap();
  bool modeIsEnabled(const std::string &mode);

  mode_v* getMode(const occa::properties &props);
  mode_v* getMode(const std::string &mode);

  device_v* newModeDevice(const occa::properties &props = occa::properties());
  kernel_v* newModeKernel(const occa::properties &props = occa::properties());
  memory_v* newModeMemory(const occa::properties &props = occa::properties());

  void freeModeDevice(device_v *dHandle);
  void freeModeKernel(kernel_v *kHandle);
  void freeModeMemory(memory_v *mHandle);

  class mode_v {
  protected:
    std::string modeName;

  public:
    std::string& name();
    virtual device_v* newDevice(const occa::properties &props = occa::properties()) = 0;
    virtual kernel_v* newKernel(const occa::properties &props = occa::properties()) = 0;
    virtual memory_v* newMemory(const occa::properties &props = occa::properties()) = 0;
  };

  template <class device_t, class kernel_t, class memory_t>
  class mode : public mode_v {
  public:
    mode(std::string modeName_) {
      modeName = modeName_;
      modeMap()[modeName] = this;
    }

    device_v* newDevice(const occa::properties &props) {
      occa::properties allProps = props;
      allProps["mode"] = modeName;
      return new device_t(allProps);
    }

    kernel_v* newKernel(const occa::properties &props) {
      occa::properties allProps = props;
      allProps["mode"] = modeName;
      return new kernel_t(allProps);
    }

    memory_v* newMemory(const occa::properties &props) {
      occa::properties allProps = props;
      allProps["mode"] = modeName;
      return new memory_t(allProps);
    }
  };
}

#endif
