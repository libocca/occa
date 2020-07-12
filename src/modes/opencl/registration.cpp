#include <occa/defines.hpp>
#include <occa/modes/opencl/utils.hpp>
#include <occa/modes/opencl/registration.hpp>

namespace occa {
  namespace opencl {
    openclMode::openclMode() :
        mode_t("OpenCL") {}

    bool openclMode::init() {
#if OCCA_OPENCL_ENABLED
      return occa::opencl::isEnabled();
#else
      return false;
#endif
    }

    styling::section& openclMode::getDescription() {
      static styling::section section("OpenCL");
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

    modeDevice_t* openclMode::newDevice(const occa::properties &props) {
      return new device(setModeProp(props));
    }

    openclMode mode;
  }
}
