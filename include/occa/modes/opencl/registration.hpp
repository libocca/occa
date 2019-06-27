#ifndef OCCA_MODES_OPENCL_REGISTRATION_HEADER
#define OCCA_MODES_OPENCL_REGISTRATION_HEADER

#include <occa/modes.hpp>
#include <occa/modes/opencl/device.hpp>
#include <occa/modes/opencl/kernel.hpp>
#include <occa/modes/opencl/memory.hpp>
#include <occa/tools/styling.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace opencl {
    class modeInfo : public modeInfo_v {
    public:
      modeInfo();

      bool init();
      styling::section& getDescription();
    };

    extern occa::mode<opencl::modeInfo,
                      opencl::device> mode;
  }
}

#endif
