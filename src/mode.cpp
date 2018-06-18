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

#include <occa/mode.hpp>
#include <occa/device.hpp>

namespace occa {
  strToModeMap& modeMap() {
    static strToModeMap modeMap_;
    return modeMap_;
  }

  void registerMode(mode_v* mode) {
    modeMap()[mode->name()] = mode;
  }

  bool modeIsEnabled(const std::string &mode) {
    return (modeMap().find(mode) != modeMap().end());
  }

  mode_v* getMode(const occa::properties &props) {
    std::string mode = props["mode"];
    const bool noMode = !mode.size();
    if (noMode || !modeIsEnabled(mode)) {
      if (noMode) {
        std::cerr << "No OCCA mode given, defaulting to [Serial] mode\n";
      } else {
        std::cerr << "[" << mode << "] mode is not enabled, defaulting to [Serial] mode\n";
      }
      mode = "Serial";
    }
    return modeMap()[mode];
  }

  device_v* newModeDevice(const occa::properties &props) {
    return getMode(props)->newDevice(props);
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
