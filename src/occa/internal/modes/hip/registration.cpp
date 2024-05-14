#include <occa/internal/utils/string.hpp>
#include <occa/internal/modes/hip/registration.hpp>
#include <occa/internal/modes/hip/utils.hpp>

namespace occa {
  namespace hip {
    hipMode::hipMode() :
        mode_t("HIP") {}

    bool hipMode::init() {
#if OCCA_HIP_ENABLED
      // Only consider hip enabled if there is an available device
      return (hip::init() && hip::getDeviceCount());
#else
      return false;
#endif
    }

    styling::section& hipMode::getDescription() {
      static styling::section section("HIP");
      if (section.size() == 0) {
        char deviceName[256];
        int deviceCount = hip::getDeviceCount();
        for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
          hipDeviceProp_t props;
          OCCA_HIP_ERROR("Getting device properties",
                         hipGetDeviceProperties(&props, deviceId));
          if (std::strlen(props.name) != 0) {
            strcpy(deviceName, props.name);
          }

          const udim_t bytes = props.totalGlobalMem;
          const std::string bytesStr = stringifyBytes(bytes);

          section
              .add("Device Name", deviceName)
              .add("Device ID"  , toString(deviceId))
              .add("Arch"       , getDeviceArch(deviceId))
              .add("Memory"     , bytesStr)
              .addDivider();
        }
        // Remove last divider
        section.groups.pop_back();
      }
      return section;
    }

    modeDevice_t* hipMode::newDevice(const occa::json &props) {
      return new device(setModeProp(props));
    }

    int hipMode::getDeviceCount(const occa::json &props) {
      return hip::getDeviceCount();
    }

    hipMode mode;
  }
}
