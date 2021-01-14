#include <occa/defines.hpp>
#include <occa/internal/modes/dpcpp/utils.hpp>
#include <occa/internal/modes/dpcpp/registration.hpp>
#include <iostream>
namespace occa {
  namespace dpcpp {
    dpcppMode::dpcppMode() :
        mode_t("dpcpp") {}

    bool dpcppMode::init() {
#if OCCA_DPCPP_ENABLED
      return occa::dpcpp::isEnabled();
#else
      return false;
#endif
    }

    styling::section& dpcppMode::getDescription() {
      static styling::section section("dpcpp");
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

    modeDevice_t* dpcppMode::newDevice(const occa::properties &props) {
      return new device(setModeProp(props));
    }

    dpcppMode mode;
  }
}
