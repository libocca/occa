#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

#include <occa/modes/metal/registration.hpp>

namespace occa {
  namespace metal {
    modeInfo::modeInfo() {}

    bool modeInfo::init() {
      return true;
    }

    styling::section& modeInfo::getDescription() {
      static styling::section section("Metal");
      if (section.size() == 0) {
        int deviceCount = 1; // getDeviceCount();
        for (int dID = 0; dID < deviceCount; ++dID) {
          udim_t bytes         = 0; //getDeviceMemorySize(pID, dID);
          std::string bytesStr = "0"; // stringifyBytes(bytes);

          section
              .add("Device Name"  , "N/A")
              .add("Device ID"    , toString(dID))
              .add("Memory"       , bytesStr)
              .addDivider();
        }
        // Remove last divider
        section.groups.pop_back();
      }
      return section;
    }

    occa::mode<metal::modeInfo,
               metal::device> mode("Metal");
  }
}

#endif
