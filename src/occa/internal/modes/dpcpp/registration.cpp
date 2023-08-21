#include <occa/defines.hpp>
#include <occa/internal/modes/dpcpp/registration.hpp>
#include <occa/internal/modes/dpcpp/device.hpp>
#include <occa/internal/modes/dpcpp/utils.hpp>
#include <occa/internal/utils/string.hpp>
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
        int platform_id{0};
        auto platform_list = ::sycl::platform::get_platforms();
        for (auto p : platform_list)
        {
          std::string platform_name_str = p.get_info<::sycl::info::platform::name>();
          section
            .add("Platform " + toString(platform_id), platform_name_str)
            .addDivider();

          int device_id{0};
          auto device_list = p.get_devices();
          for (auto d : device_list)
          {
            std::string device_type_str;
            if (d.is_gpu())
            {
              device_type_str = "gpu";
            }
            else if (d.is_cpu())
            {
              device_type_str = "cpu";
            }
            else if (d.is_accelerator())
            {
              device_type_str = "accelerator";
            }
            else
            {
              device_type_str = "TYPE UNKNOWN";
            }

            std::string device_name_str = d.get_info<::sycl::info::device::name>();

            uint32_t max_compute_units = d.get_info<::sycl::info::device::max_compute_units>();

            // Global memory is returned in bytes
            uint64_t global_memory_B = d.get_info<::sycl::info::device::global_mem_size>();
            std::string global_memory_str = stringifyBytes(global_memory_B);

            // Local memory is returned in bytes
            uint64_t local_memory_B = d.get_info<::sycl::info::device::local_mem_size>();
            std::string local_memory_str = stringifyBytes(local_memory_B);

            section
              .add("Device " + toString(device_id), device_name_str)
              .add("Device Type", device_type_str)
              .add("Compute Cores", toString(max_compute_units))
              .add("Global Memory", global_memory_str)
              .add("Local Memory", local_memory_str)
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

    modeDevice_t* dpcppMode::newDevice(const occa::json &props) {
      // Refactor this into a helper function.
      OCCA_ERROR(
          "[dpcpp] device not given a [platform_id] integer",
          props.has("platform_id") && props["platform_id"].isNumber());
      int platformID = props.get<int>("platform_id");
      
      auto platforms{::sycl::platform::get_platforms()};
      OCCA_ERROR(
          "Invalid platform number (" + toString(platformID) + ")",
          (static_cast<size_t>(platformID) < platforms.size()));
      auto& platform = platforms[platformID];

      OCCA_ERROR(
          "[dpcpp] device not given a [device_id] integer",
          props.has("device_id") && props["device_id"].isNumber());

      int deviceID = props.get<int>("device_id");
      auto devices{platform.get_devices()};
      OCCA_ERROR(
          "Invalid device number (" + toString(deviceID) + ")",
          (static_cast<size_t>(deviceID) < devices.size()));
      auto& dpcppDevice = devices[deviceID];

      return new occa::dpcpp::device(setModeProp(props), dpcppDevice);
    }

    int dpcppMode::getDeviceCount(const occa::json& props) {
       OCCA_ERROR(
            "[dpcppMode::getDeviceCount] not given a [platform_id] integer",
            props.has("platform_id") && props["platform_id"].isNumber());

      int pID{props.get<int>("platform_id")};

      return static_cast<int>(::sycl::platform::get_platforms()[pID].get_devices().size());
    }

    dpcppMode mode;
  }
}
