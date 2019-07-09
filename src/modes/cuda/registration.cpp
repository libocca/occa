#include <occa/defines.hpp>
#include <occa/modes/cuda/registration.hpp>
#include <occa/modes/cuda/utils.hpp>

namespace occa {
  namespace cuda {
    modeInfo::modeInfo() {}

    bool modeInfo::init() {
#if OCCA_CUDA_ENABLED
      // Only consider cuda enabled if there is an available device
      return (cuda::init() && cuda::getDeviceCount());
#else
      return false;
#endif
    }

    styling::section& modeInfo::getDescription() {
      static styling::section section("CUDA");
      if (section.size() == 0) {
        char deviceName[1024];
        int deviceCount = cuda::getDeviceCount();
        for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
          const udim_t bytes         = getDeviceMemorySize(getDevice(deviceId));
          const std::string bytesStr = stringifyBytes(bytes);

          OCCA_CUDA_ERROR("Getting Device Name",
                          cuDeviceGetName(deviceName, 1024, deviceId));

          section
            .add("Device Name", deviceName)
            .add("Device ID"  , toString(deviceId))
            .add("Memory"     , bytesStr)
            .addDivider();
        }
        // Remove last divider
        section.groups.pop_back();
      }
      return section;
    }

    occa::mode<cuda::modeInfo,
               cuda::device> mode("CUDA");
  }
}
