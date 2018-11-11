#include <occa/defines.hpp>

#if OCCA_HIP_ENABLED

#include <occa/mode/hip/registration.hpp>
#include <occa/mode/hip/utils.hpp>

namespace occa {
  namespace hip {
    modeInfo::modeInfo() {}

    bool modeInfo::init() {
      return hip::init();
    }

    styling::section& modeInfo::getDescription() {
      static styling::section section("HIP");
      if (section.size() == 0) {
        hipDevice_t device;
        char deviceName[256];
        int deviceCount = hip::getDeviceCount();
        for (int i = 0; i < deviceCount; ++i) {
          hipDeviceProp_t props;
          OCCA_HIP_ERROR("Getting device properties",
                         hipGetDeviceProperties(&props, i));
          strcpy(deviceName, props.name);

          const udim_t bytes         = props.totalGlobalMem;
          const std::string bytesStr = stringifyBytes(bytes);

          section
            .add("Device ID"  ,  toString(i))
            .add("Arch"       , "gfx" + toString(props.gcnArch))
            .add("Device Name",  deviceName)
            .add("Memory"     ,  bytesStr)
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

#endif
