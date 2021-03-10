#ifndef OCCA_INTERNAL_MODE_HEADER
#define OCCA_INTERNAL_MODE_HEADER

#include <iostream>
#include <map>

#include <occa/defines.hpp>
#include <occa/types/json.hpp>
#include <occa/internal/utils/styling.hpp>
#include <occa/core/device.hpp>

namespace occa {
  class modeDevice_t;
  class mode_t;

  typedef std::map<std::string, mode_t*> strToModeMap;

  class mode_t {
   protected:
    std::string modeName;

   public:
    mode_t(const std::string &modeName_);

    std::string& name();

    occa::json setModeProp(const occa::json &props);

    virtual bool init() = 0;

    virtual styling::section &getDescription();

    virtual modeDevice_t* newDevice(const occa::json &props) = 0;

    virtual int getDeviceCount(const occa::json &props) = 0;
  };

  strToModeMap& getUnsafeModeMap();

  strToModeMap& getModeMap();

  void registerMode(mode_t* mode);

  void initializeModes();

  mode_t* getMode(const std::string &mode);

  mode_t* getModeFromProps(const occa::json &props);

  modeDevice_t* newModeDevice(const occa::json &props);
}

#endif
