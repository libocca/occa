#ifndef OCCA_INTERNAL_MODES_HIP_REGISTRATION_HEADER
#define OCCA_INTERNAL_MODES_HIP_REGISTRATION_HEADER

#include <occa/internal/modes.hpp>
#include <occa/internal/modes/hip/device.hpp>
#include <occa/internal/modes/hip/kernel.hpp>
#include <occa/internal/modes/hip/memory.hpp>
#include <occa/internal/utils/styling.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace hip {
    class hipMode : public mode_t {
    public:
      hipMode();

      bool init();

      styling::section& getDescription();

      modeDevice_t* newDevice(const occa::json &props);

      int getDeviceCount(const occa::json &props);
    };

    extern hipMode mode;
  }
}

#endif
