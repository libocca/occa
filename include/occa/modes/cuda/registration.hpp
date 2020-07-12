#include <occa/defines.hpp>

#ifndef OCCA_MODES_CUDA_REGISTRATION_HEADER
#define OCCA_MODES_CUDA_REGISTRATION_HEADER

#include <occa/modes.hpp>
#include <occa/modes/cuda/device.hpp>
#include <occa/modes/cuda/kernel.hpp>
#include <occa/modes/cuda/memory.hpp>
#include <occa/tools/styling.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace cuda {
    class cudaMode : public mode_t {
    public:
      cudaMode();

      bool init();

      styling::section& getDescription();

      modeDevice_t* newDevice(const occa::properties &props);
    };

    extern cudaMode mode;
  }
}

#endif
