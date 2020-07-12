#include <occa/core/device.hpp>
#include <occa/io/output.hpp>
#include <occa/modes.hpp>

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

  occa::properties mode_t::setModeProp(const occa::properties &props) {
    occa::properties propsWithMode = props;
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
    getUnsafeModeMap()[mode->name()] = mode;
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

  bool modeIsEnabled(const std::string &mode) {
    strToModeMap &modeMap = getModeMap();
    return (modeMap.find(mode) != modeMap.end());
  }

  mode_t* getMode(const occa::properties &props) {
    std::string mode = props["mode"];
    const bool noMode = !mode.size();
    if (noMode || !modeIsEnabled(mode)) {
      if (noMode) {
        io::stderr << "No OCCA mode given, defaulting to [Serial] mode\n";
      } else {
        io::stderr << "[" << mode << "] mode is not enabled, defaulting to [Serial] mode\n";
      }
      mode = "Serial";
    }
    return getModeMap()[mode];
  }

  modeDevice_t* newModeDevice(const occa::properties &props) {
    return getMode(props)->newDevice(props);
  }
}
