#include <occa/defines.hpp>

#if OCCA_OPENCL_ENABLED

#include <occa/mode/opencl/utils.hpp>
#include <occa/mode/opencl/registration.hpp>

namespace occa {
  namespace opencl {
    modeInfo::modeInfo() {}

    bool modeInfo::init() {
      return occa::opencl::isEnabled();
    }

    styling::section& modeInfo::getDescription() {
      static styling::section section("OpenCL");
      if (section.size() == 0) {
        int platformCount = getPlatformCount();
        for (int pID = 0; pID < platformCount; ++pID) {
          int deviceCount = getDeviceCountInPlatform(pID);
          for (int dID = 0; dID < deviceCount; ++dID) {
            udim_t bytes         = getDeviceMemorySize(pID, dID);
            std::string bytesStr = stringifyBytes(bytes);

            section
              .add("Device Name"  , deviceName(pID, dID))
              .add("Driver Vendor", info::vendor(deviceVendor(pID,dID)))
              .add("Platform ID"  , toString(pID))
              .add("Device ID"    , toString(dID))
              .add("Memory"       , bytesStr)
              .addDivider();
          }
        }
        // Remove last divider
        section.groups.pop_back();
      }
      return section;
    }

    occa::mode<opencl::modeInfo,
               opencl::device> mode("OpenCL");
  }
}

#endif
