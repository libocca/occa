#ifndef OCCA_MODES_OPENMP_REGISTRATION_HEADER
#define OCCA_MODES_OPENMP_REGISTRATION_HEADER

#include <occa/modes.hpp>
#include <occa/modes/openmp/device.hpp>
#include <occa/modes/serial/memory.hpp>
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

#endif
