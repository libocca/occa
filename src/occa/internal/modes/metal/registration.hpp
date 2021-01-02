#ifndef OCCA_INTERNAL_MODES_METAL_REGISTRATION_HEADER
#define OCCA_INTERNAL_MODES_METAL_REGISTRATION_HEADER

#include <occa/internal/modes.hpp>
#include <occa/internal/modes/metal/device.hpp>
#include <occa/internal/modes/metal/kernel.hpp>
#include <occa/internal/modes/metal/memory.hpp>
#include <occa/internal/utils/styling.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace metal {
    class metalMode : public mode_t {
    public:
      metalMode();

      bool init();

      styling::section& getDescription();

      modeDevice_t* newDevice(const occa::json &props);

      int getDeviceCount(const occa::json &props);
    };

    extern metalMode mode;
  }
}

#endif
