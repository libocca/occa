#ifndef OCCA_MODES_METAL_REGISTRATION_HEADER
#define OCCA_MODES_METAL_REGISTRATION_HEADER

#include <occa/modes.hpp>
#include <occa/modes/metal/device.hpp>
#include <occa/modes/metal/kernel.hpp>
#include <occa/modes/metal/memory.hpp>
#include <occa/tools/styling.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace metal {
    class metalMode : public mode_t {
    public:
      metalMode();

      bool init();

      styling::section& getDescription();

      modeDevice_t* newDevice(const occa::properties &props);
    };

    extern metalMode mode;
  }
}

#endif
