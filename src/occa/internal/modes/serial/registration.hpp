#ifndef OCCA_INTERNAL_MODES_SERIAL_REGISTRATION_HEADER
#define OCCA_INTERNAL_MODES_SERIAL_REGISTRATION_HEADER

#include <occa/defines.hpp>
#include <occa/internal/modes.hpp>
#include <occa/internal/modes/serial/device.hpp>
#include <occa/internal/modes/serial/kernel.hpp>
#include <occa/internal/modes/serial/memory.hpp>
#include <occa/internal/utils/styling.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace serial {
    class serialMode : public mode_t {
    public:
      serialMode();

      bool init();

      styling::section& getDescription();

      modeDevice_t* newDevice(const occa::json &props);

      int getDeviceCount(const occa::json &props);
    };

    extern serialMode mode;
  }
}

#endif
