#include <occa/defines.hpp>
#include <occa/modes/oneapi/utils.hpp>
#include <occa/modes/oneapi/registration.hpp>

namespace occa {
  namespace oneapi {
    oneapiMode::oneapiMode() :
        mode_t("oneAPI") {}

    bool oneapiMode::init() {
#if OCCA_ONEAPI_ENABLED
      return occa::oneapi::isEnabled();
#else
      return false;
#endif
    }

    styling::section& oneapiMode::getDescription() {
      static styling::section section("oneAPI");
      if (section.size() == 0) {
        int platformCount = getPlatformCount();
        for (int platformId = 0; platformId < platformCount; ++platformId) {
          int deviceCount = getDeviceCountInPlatform(platformId);
          for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
            udim_t bytes = getDeviceMemorySize(platformId, deviceId);
            std::string bytesStr = stringifyBytes(bytes);

            section
              .add("Device Name"  , deviceName(platformId, deviceId))
              .add("Driver Vendor", info::vendor(deviceVendor(platformId, deviceId)))
              .add("Platform ID"  , toString(platformId))
              .add("Device ID"    , toString(deviceId))
              .add("Memory"       , bytesStr)
              .addDivider();
          }
        }
        // Remove last divider
        section.groups.pop_back();
      }
      return section;
    }

    modeDevice_t* oneapiMode::newDevice(const occa::properties &props) {
      return new device(setModeProp(props));
    }

    oneapiMode mode;
  }
}
