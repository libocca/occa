#include <occa/core/base.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/io.hpp>
#include <occa/internal/modes/serial/kernel.hpp>
#include <occa/internal/lang/modes/serial.hpp>

namespace occa {
  namespace serial {
    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   const occa::json &properties_) :
      occa::modeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
      dlHandle(NULL),
      function(NULL),
      isLauncherKernel(false) {}

    kernel::~kernel() {
      if (dlHandle) {
        sys::dlclose(dlHandle);
        dlHandle = NULL;
      }
    }

    int kernel::maxDims() const {
      return 3;
    }

    dim kernel::maxOuterDims() const {
      return dim(occa::UDIM_DEFAULT, occa::UDIM_DEFAULT, occa::UDIM_DEFAULT);
    }

    dim kernel::maxInnerDims() const {
      return dim(occa::UDIM_DEFAULT, occa::UDIM_DEFAULT, occa::UDIM_DEFAULT);
    }

    const lang::kernelMetadata_t& kernel::getMetadata() const {
      return metadata;
    }

    void kernel::run() const {
      const int args = (int) arguments.size();
      if (!args) {
        vArgs.resize(1);
      } else if ((int) vArgs.size() < args) {
        vArgs.resize(args);
      }

      // Set arguments
      for (int i = 0; i < args; ++i) {
        vArgs[i] = arguments[i].ptr();
      }

      sys::runFunction(function, args, &(vArgs[0]));
    }
  }
}
