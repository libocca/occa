#include <occa/core/device.hpp>
#include <occa/io/output.hpp>
#include <occa/mode.hpp>

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
        io::stderr << "No OCCA mode given, defaulting to [Serial] mode\n";
      } else {
        io::stderr << "[" << mode << "] mode is not enabled, defaulting to [Serial] mode\n";
      }
      mode = "Serial";
    }
    return modeMap()[mode];
  }

  modeDevice_t* newModeDevice(const occa::properties &props) {
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
