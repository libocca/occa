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
