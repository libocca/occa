/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
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

#include "occa/mode.hpp"
#include "occa/device.hpp"
#include "occa/kernel.hpp"
#include "occa/memory.hpp"

namespace occa {
  strToModeMap_t& modeMap() {
    static strToModeMap_t modeMap_;
    return modeMap_;
  }

  void registerMode(mode_v* mode) {
    modeMap()[mode->name()] = mode;
  }

  bool modeIsEnabled(const std::string &mode) {
    return (modeMap().find(mode) != modeMap().end());
  }

  mode_v* getMode(const occa::properties &props) {
    std::string mode = props["mode"].string();
    const bool noMode = !mode.size();
    if (noMode || !modeIsEnabled(mode)) {
      if (noMode) {
        std::cout << "No OCCA mode given, defaulting to [Serial] mode\n";
      } else {
        std::cout << "Mode [" << mode << "] is not enabled, defaulting to [Serial] mode\n";
      }
      return modeMap()["Serial"];
    }
    return modeMap()[props["mode"]];
  }

  device_v* newModeDevice(const occa::properties &props) {
    return getMode(props)->newDevice(props);
  }

  kernel_v* newModeKernel(const occa::properties &props) {
    return getMode(props)->newKernel(props);
  }

  memory_v* newModeMemory(const occa::properties &props) {
    return getMode(props)->newMemory(props);
  }

  void freeModeDevice(device_v *dHandle) {
    delete dHandle;
  }

  void freeModeKernel(kernel_v *kHandle) {
    delete kHandle;
  }

  void freeModeMemory(memory_v *mHandle) {
    delete mHandle;
  }

  modeInfo_v::modeInfo_v() {}

  styling::section& modeInfo_v::getDescription() {
    static styling::section section;
    return section;
  }

  std::string& mode_v::name() {
    return modeName;
  }
}
