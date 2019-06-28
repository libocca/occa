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
        int deviceCount = metalDevice_t::getCount();
        for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
          metalDevice_t device = metalDevice_t::fromId(deviceId);

          udim_t bytes = device.getMemorySize();
          std::string bytesStr = stringifyBytes(bytes);

          section
              .add("Device Name", device.getName())
              .add("Device ID"  , toString(deviceId))
              .add("Memory"     , bytesStr)
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
