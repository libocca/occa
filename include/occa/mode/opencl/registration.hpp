#include <occa/defines.hpp>

#if OCCA_OPENCL_ENABLED
#  ifndef OCCA_MODES_OPENCL_REGISTRATION_HEADER
#  define OCCA_MODES_OPENCL_REGISTRATION_HEADER

#include <occa/mode.hpp>
#include <occa/mode/opencl/device.hpp>
#include <occa/mode/opencl/kernel.hpp>
#include <occa/mode/opencl/memory.hpp>
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

#  endif
#endif
