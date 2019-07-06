#include <occa/defines.hpp>
#include <occa/modes/opencl/utils.hpp>
#include <occa/modes/opencl/registration.hpp>

namespace occa {
  namespace opencl {
    modeInfo::modeInfo() {}

    bool modeInfo::init() {
#if OCCA_OPENCL_ENABLED
      return occa::opencl::isEnabled();
#else
      return false;
#endif
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
