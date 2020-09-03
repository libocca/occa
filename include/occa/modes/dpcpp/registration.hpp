#ifndef OCCA_MODES_ONEAPI_REGISTRATION_HEADER
#define OCCA_MODES_ONEAPI_REGISTRATION_HEADER

#include <occa/modes.hpp>
#include <occa/modes/oneapi/device.hpp>
#include <occa/modes/oneapi/kernel.hpp>
#include <occa/modes/oneapi/memory.hpp>
#include <occa/tools/styling.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace oneapi {
    class oneapiMode : public mode_t {
    public:
      oneapiMode();

      bool init();

      styling::section& getDescription();

      modeDevice_t* newDevice(const occa::properties &props);
    };

    extern oneapiMode mode;
  }
}

#endif
