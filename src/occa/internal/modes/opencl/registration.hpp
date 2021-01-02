#ifndef OCCA_INTERNAL_MODES_OPENCL_REGISTRATION_HEADER
#define OCCA_INTERNAL_MODES_OPENCL_REGISTRATION_HEADER

#include <occa/internal/modes.hpp>
#include <occa/internal/modes/opencl/device.hpp>
#include <occa/internal/modes/opencl/kernel.hpp>
#include <occa/internal/modes/opencl/memory.hpp>
#include <occa/internal/utils/styling.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace opencl {
    class openclMode : public mode_t {
    public:
      openclMode();

      bool init();

      styling::section& getDescription();

      modeDevice_t* newDevice(const occa::json &props);

      int getDeviceCount(const occa::json &props);
    };

    extern openclMode mode;
  }
}

#endif
