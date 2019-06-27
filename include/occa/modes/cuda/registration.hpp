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
    class modeInfo : public modeInfo_v {
    public:
      modeInfo();

      bool init();
      styling::section& getDescription();
    };

    extern occa::mode<cuda::modeInfo,
                      cuda::device> mode;
  }
}

#endif
