#ifndef OCCA_MODES_SERIAL_REGISTRATION_HEADER
#define OCCA_MODES_SERIAL_REGISTRATION_HEADER

#include <occa/defines.hpp>
#include <occa/mode.hpp>
#include <occa/mode/serial/device.hpp>
#include <occa/mode/serial/kernel.hpp>
#include <occa/mode/serial/memory.hpp>
#include <occa/tools/styling.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace serial {
    class modeInfo : public modeInfo_v {
    public:
      modeInfo();

      bool init();
      styling::section& getDescription();
    };

    extern occa::mode<serial::modeInfo,
                      serial::device> mode;
  }
}

#endif
