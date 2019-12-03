#include <occa/modes/hip/registration.hpp>
#include <occa/modes/hip/utils.hpp>

namespace occa {
  namespace hip {
    modeInfo::modeInfo() {}

    bool modeInfo::init() {
#if OCCA_HIP_ENABLED
      // Only consider hip enabled if there is an available device
      return (hip::init() && hip::getDeviceCount());
#else
      return false;
#endif
    }

    styling::section& modeInfo::getDescription() {
      static styling::section section("HIP");
      if (section.size() == 0) {
        char deviceName[256];
        int deviceCount = hip::getDeviceCount();
        for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
          hipDeviceProp_t props;
          OCCA_HIP_ERROR("Getting device properties",
                         hipGetDeviceProperties(&props, deviceId));
          strcpy(deviceName, props.name);

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

    occa::mode<hip::modeInfo,
               hip::device> mode("HIP");
  }
}
