#include <occa/defines.hpp>
#include <occa/internal/utils/string.hpp>
#include <occa/internal/modes/opencl/utils.hpp>
#include <occa/internal/modes/opencl/registration.hpp>

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
          std::string platform_name_str = platformName(platformId);
          section
            .add("Platform " + toString(platformId), platform_name_str)
            .addDivider();

          int deviceCount = getDeviceCountInPlatform(platformId);
          for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
            std::string device_name_str = deviceName(platformId, deviceId);
            info::device_type type = deviceType(platformId, deviceId);
            std::string device_type_str;
            switch (type) {
              case info::device_type::cpu:
                device_type_str = "cpu";
                break;
              case info::device_type::gpu:
                device_type_str = "gpu";
                break;
              case info::device_type::accelerator:
                device_type_str = "accelerator";
                break;
              case info::device_type::all:
                device_type_str = "all!?";
                break;
            }

            int compute_cores = deviceCoreCount(platformId, deviceId);
            udim_t global_memory_B = deviceGlobalMemSize(platformId, deviceId);
            std::string global_memory_str = stringifyBytes(global_memory_B);

            section
              .add("Device " + toString(deviceId), device_name_str)
              .add("Device Type", device_type_str)
              .add("Compute Cores", toString(compute_cores))
              .add("Global Memory", global_memory_str)
              .addDivider();
          }
        }
        // Remove last divider
        section.groups.pop_back();
      }
      return section;
    }

    modeDevice_t* openclMode::newDevice(const occa::json &props) {
      return new device(setModeProp(props));
    }

    int openclMode::getDeviceCount(const occa::json &props) {
      OCCA_ERROR("[OpenCL] getDeviceCount not given a [platform_id] integer",
                 props.has("platform_id") &&
                 props["platform_id"].isNumber());

      int platformId = props.get<int>("platform_id");

      return getDeviceCountInPlatform(platformId);
    }

    openclMode mode;
  }
}
