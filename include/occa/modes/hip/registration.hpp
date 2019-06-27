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
    class modeInfo : public modeInfo_v {
    public:
      modeInfo();

      bool init();
      styling::section& getDescription();
    };

    extern occa::mode<hip::modeInfo,
                      hip::device> mode;
  }
}

#endif
