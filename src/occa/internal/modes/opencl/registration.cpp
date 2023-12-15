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
      static styling::section section(modeName);
      if (section.size() == 0) {
        int platform_id{0};
        auto platform_list = getPlatforms();
        for (auto& p : platform_list) {
          std::string platform_name_str = platformName(p);
          section
            .add("Platform " + toString(platform_id), platform_name_str)
            .addDivider();

          int device_id{0};
          auto device_list = getDevicesInPlatform(p);
          for (auto& d : device_list) {
            std::string device_name_str = deviceName(d);
            cl_device_type device_type = deviceType(d);
            std::string device_type_str;
            switch (device_type) {
              case CL_DEVICE_TYPE_CPU:
                device_type_str = "cpu";
                break;
              case CL_DEVICE_TYPE_GPU:
                device_type_str = "gpu";
                break;
              case CL_DEVICE_TYPE_ACCELERATOR:
                device_type_str = "accelerator";
                break;
              case CL_DEVICE_TYPE_ALL:
                device_type_str = "all";
                break;
              default:
                device_type_str = "???";
                break;
            }

            int compute_cores = deviceCoreCount(d);
            udim_t global_memory_B = deviceGlobalMemSize(d);
            std::string global_memory_str = stringifyBytes(global_memory_B);

            section
              .add("Device " + toString(device_id), device_name_str)
              .add("Device Type", device_type_str)
              .add("Compute Cores", toString(compute_cores))
              .add("Global Memory", global_memory_str)
              .addDivider();

            ++device_id;
          }
          ++platform_id;
        }
        // Remove last divider
        section.groups.pop_back();
      }
      return section;
    }

    modeDevice_t* openclMode::newDevice(const occa::json& properties) {
      OCCA_ERROR("[OpenCL] device not given a [platform_id] integer",
                 properties.has("platform_id") &&
                 properties["platform_id"].isNumber());
      int platform_id = properties.get<int>("platform_id");

      auto platforms{getPlatforms()};
      OCCA_ERROR("Invalid platform number (" + toString(platform_id) + ")",
        (static_cast<size_t>(platform_id) < platforms.size()));
      auto& platform = platforms[platform_id];

      OCCA_ERROR("[OpenCL] device not given a [device_id] integer",
                 properties.has("device_id") &&
                 properties["device_id"].isNumber());
      int device_id = properties.get<int>("device_id");

      auto devices{getDevicesInPlatform(platform)};
      OCCA_ERROR("Invalid device number (" + toString(device_id) + ")",
          (static_cast<size_t>(device_id) < devices.size()));
      auto& opencl_device = devices[device_id]; 
      
      return new device(setModeProp(properties), opencl_device);
    }

    int openclMode::getDeviceCount(const occa::json& properties) {
      OCCA_ERROR("[OpenCL] getDeviceCount not given a [platform_id] integer",
                 properties.has("platform_id") && properties["platform_id"].isNumber());
      int platform_id = properties.get<int>("platform_id");

      auto platforms{getPlatforms()};
      OCCA_ERROR("Invalid platform number (" + toString(platform_id) + ")",
        (static_cast<size_t>(platform_id) < platforms.size()));
      auto& platform = platforms[platform_id];

      return getDeviceCountInPlatform(platform);
    }

    openclMode mode;
  }
}
