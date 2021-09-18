#include <occa/defines.hpp>

#ifndef OCCA_MODES_DPCPP_REGISTRATION_HEADER
#define OCCA_MODES_DPCPP_REGISTRATION_HEADER

#include <occa/internal/modes.hpp>
#include <occa/internal/utils/styling.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace dpcpp {
    class dpcppMode : public mode_t {
    public:
      dpcppMode();

      virtual bool init() override;

      virtual styling::section& getDescription() override;

      virtual modeDevice_t* newDevice(const occa::json &props) override;

      virtual int getDeviceCount(const occa::json &props) override;
    };

    extern dpcppMode mode;
  }
}

#endif
