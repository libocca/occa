#ifndef OCCA_MODE_HEADER
#define OCCA_MODE_HEADER

#include <iostream>
#include <map>

#include <occa/defines.hpp>
#include <occa/tools/properties.hpp>
#include <occa/tools/styling.hpp>
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

    occa::properties setModeProp(const occa::properties &props);

    virtual bool init() = 0;

    virtual styling::section &getDescription();

    virtual modeDevice_t* newDevice(const occa::properties &props) = 0;
  };

  strToModeMap& getUnsafeModeMap();

  strToModeMap& getModeMap();

  void registerMode(mode_t* mode);

  void initializeModes();

  bool modeIsEnabled(const std::string &mode);

  mode_t* getMode(const std::string &mode);

  mode_t* getModeFromProps(const occa::properties &props);

  modeDevice_t* newModeDevice(const occa::properties &props);
}

#endif
