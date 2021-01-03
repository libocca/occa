#ifndef OCCA_INTERNAL_MODES_OPENMP_REGISTRATION_HEADER
#define OCCA_INTERNAL_MODES_OPENMP_REGISTRATION_HEADER

#include <occa/internal/modes.hpp>
#include <occa/internal/modes/openmp/device.hpp>
#include <occa/internal/modes/serial/memory.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace openmp {
    class openmpMode : public mode_t {
    public:
      openmpMode();

      bool init();

      void setupProperties();

      modeDevice_t* newDevice(const occa::json &props);

      int getDeviceCount(const occa::json &props);
    };

    extern openmpMode mode;
  }
}

#endif
