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

#ifndef OCCA_MODE_HEADER
#define OCCA_MODE_HEADER

#include <iostream>
#include <map>

#include <occa/defines.hpp>
#include <occa/tools/properties.hpp>
#include <occa/tools/styling.hpp>
#include <occa/core/device.hpp>

namespace occa {
  class modeDevice_t; class device;

  class mode_v;

  typedef std::map<std::string, mode_v*> strToModeMap;
  typedef strToModeMap::iterator         strToModeMapIterator;

  strToModeMap& modeMap();
  void registerMode(mode_v* mode);
  bool modeIsEnabled(const std::string &mode);

  mode_v* getMode(const occa::properties &props);

  modeDevice_t* newModeDevice(const occa::properties &props = occa::properties());

  class modeInfo_v {
  public:
    modeInfo_v();

    virtual bool init() = 0;
    virtual styling::section& getDescription();
  };

  class mode_v {
  protected:
    std::string modeName;

  public:
    std::string& name();
    virtual styling::section &getDescription() = 0;
    virtual modeDevice_t* newDevice(const occa::properties &props = occa::properties()) = 0;
  };

  template <class modeInfo_t,
            class device_t>
  class mode : public mode_v {
  public:
    mode(std::string modeName_) {
      modeName = modeName_;
      if (modeInfo_t().init()) {
        registerMode(this);
      }
    }

    styling::section &getDescription() {
      return modeInfo_t().getDescription();
    }

    modeDevice_t* newDevice(const occa::properties &props) {
      occa::properties allProps = props;
      allProps["mode"] = modeName;
      return new device_t(allProps);
    }
  };
}

#endif
