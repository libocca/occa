#ifndef OCCA_MODES_OPENMP_REGISTRATION_HEADER
#define OCCA_MODES_OPENMP_REGISTRATION_HEADER

#include <occa/modes.hpp>
#include <occa/modes/openmp/device.hpp>
#include <occa/modes/serial/memory.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace openmp {
    class openmpMode : public mode_t {
    public:
      openmpMode();

      bool init();

      void setupProperties();

      modeDevice_t* newDevice(const occa::properties &props);
    };

    extern openmpMode mode;
  }
}

#endif
