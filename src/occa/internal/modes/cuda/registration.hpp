#include <occa/defines.hpp>

#ifndef OCCA_INTERNAL_MODES_CUDA_REGISTRATION_HEADER
#define OCCA_INTERNAL_MODES_CUDA_REGISTRATION_HEADER

#include <occa/internal/modes.hpp>
#include <occa/internal/modes/cuda/device.hpp>
#include <occa/internal/modes/cuda/kernel.hpp>
#include <occa/internal/modes/cuda/memory.hpp>
#include <occa/internal/utils/styling.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace cuda {
    class cudaMode : public mode_t {
    public:
      cudaMode();

      bool init();

      styling::section& getDescription();

      modeDevice_t* newDevice(const occa::json &props);

      int getDeviceCount(const occa::json &props);
    };

    extern cudaMode mode;
  }
}

#endif
