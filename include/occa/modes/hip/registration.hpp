#ifndef OCCA_MODES_HIP_REGISTRATION_HEADER
#define OCCA_MODES_HIP_REGISTRATION_HEADER

#include <occa/modes.hpp>
#include <occa/modes/hip/device.hpp>
#include <occa/modes/hip/kernel.hpp>
#include <occa/modes/hip/memory.hpp>
#include <occa/tools/styling.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace hip {
    class hipMode : public mode_t {
    public:
      hipMode();

      bool init();

      styling::section& getDescription();

      modeDevice_t* newDevice(const occa::properties &props);
    };

    extern hipMode mode;
  }
}

#endif
