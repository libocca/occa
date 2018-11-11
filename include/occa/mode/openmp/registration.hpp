#include <occa/defines.hpp>

#if OCCA_OPENMP_ENABLED
#  ifndef OCCA_MODES_OPENMP_REGISTRATION_HEADER
#  define OCCA_MODES_OPENMP_REGISTRATION_HEADER

#include <occa/mode.hpp>
#include <occa/mode/openmp/device.hpp>
#include <occa/mode/serial/memory.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace openmp {
    class modeInfo : public modeInfo_v {
    public:
      modeInfo();

      bool init();
      void setupProperties();
    };

    extern occa::mode<openmp::modeInfo,
                      openmp::device> mode;
  }
}

#  endif
#endif
