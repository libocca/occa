#include <occa/defines.hpp>

#ifndef OCCA_MODES_DPCPP_REGISTRATION_HEADER
#define OCCA_MODES_DPCPP_REGISTRATION_HEADER

#include <occa/internal/modes.hpp>
#include <occa/internal/modes/dpcpp/device.hpp>
#include <occa/internal/modes/dpcpp/kernel.hpp>
#include <occa/internal/modes/dpcpp/memory.hpp>
#include <occa/internal/utils/styling.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace dpcpp {
    class dpcppMode : public mode_t {
    public:
      dpcppMode();

      bool init();

      styling::section& getDescription();

      modeDevice_t* newDevice(const occa::properties &props);
    };

    extern dpcppMode mode;
  }
}

#endif
