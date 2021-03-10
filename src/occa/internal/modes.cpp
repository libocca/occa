#include <occa/core/device.hpp>
#include <occa/internal/io/output.hpp>
#include <occa/internal/modes.hpp>
#include <occa/internal/utils/string.hpp>

namespace occa {
  //---[ mode_t ]-----------------------
  mode_t::mode_t(const std::string &modeName_) :
    modeName(modeName_) {
    registerMode(this);
  }

  std::string& mode_t::name() {
    return modeName;
  }

  styling::section& mode_t::getDescription() {
    static styling::section section;
    return section;
  }

  occa::json mode_t::setModeProp(const occa::json &props) {
    occa::json propsWithMode = props;
    propsWithMode["mode"] = modeName;
    return propsWithMode;
  }
  //====================================

  strToModeMap& getUnsafeModeMap() {
    static strToModeMap modeMap;
    return modeMap;
  }

  strToModeMap& getModeMap() {
    initializeModes();
    return getUnsafeModeMap();
  }

  void registerMode(mode_t* mode) {
    const std::string caseInsensitiveMode = lowercase(mode->name());
    getUnsafeModeMap()[caseInsensitiveMode] = mode;
  }

  void initializeModes() {
    static bool isInitialized = false;

    if (isInitialized) {
      return;
    }

    strToModeMap &modeMap = getUnsafeModeMap();
    strToModeMap::iterator it = modeMap.begin();

    strToModeMap validModeMap;

    for (; it != modeMap.end(); ++it) {
      const std::string &modeName = it->first;
      mode_t *mode = it->second;

      if (mode->init()) {
        validModeMap[modeName] = mode;
      }
    }

    modeMap.swap(validModeMap);
    isInitialized = true;
  }

  mode_t* getMode(const std::string &mode) {
    const std::string caseInsensitiveMode = lowercase(mode);
    strToModeMap &modeMap = getModeMap();

    strToModeMap::iterator it = modeMap.find(caseInsensitiveMode);
    if (it != modeMap.end()) {
      return it->second;
    }

    return NULL;
  }

  mode_t* getModeFromProps(const occa::json &props) {
    std::string modeName = props["mode"];
    mode_t *mode = getMode(modeName);

    if (mode) {
      return mode;
    }

    if (modeName.size()) {
      io::stderr << "[" << modeName << "] mode is not enabled, defaulting to [Serial] mode\n";
    } else {
      io::stderr << "No OCCA mode given, defaulting to [Serial] mode\n";
    }
    return getMode("Serial");
  }

  modeDevice_t* newModeDevice(const occa::json &props) {
    return getModeFromProps(props)->newDevice(props);
  }
}
