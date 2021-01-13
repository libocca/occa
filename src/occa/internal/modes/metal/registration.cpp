#include <occa/defines.hpp>
#include <occa/internal/utils/string.hpp>
#include <occa/internal/modes/metal/registration.hpp>

namespace occa {
  namespace metal {
    metalMode::metalMode() :
        mode_t("Metal") {}

    bool metalMode::init() {
#if OCCA_METAL_ENABLED
      // Only consider metal enabled if there is an available device
      return api::metal::getDeviceCount();
#else
      return false;
#endif
    }

    styling::section& metalMode::getDescription() {
      static styling::section section("Metal");
      if (section.size() == 0) {
        int deviceCount = api::metal::getDeviceCount();
        for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
          api::metal::device_t device = api::metal::getDevice(deviceId);

          udim_t bytes = device.getMemorySize();
          std::string bytesStr = stringifyBytes(bytes);

          section
              .add("Device Name", device.getName())
              .add("Device ID"  , toString(deviceId))
              .add("Memory"     , bytesStr)
              .addDivider();
        }
        // Remove last divider
        section.groups.pop_back();
      }
      return section;
    }

    modeDevice_t* metalMode::newDevice(const occa::json &props) {
      return new device(setModeProp(props));
    }

    int metalMode::getDeviceCount(const occa::json &props) {
      return api::metal::getDeviceCount();
    }

    metalMode mode;
  }
}
