#include <occa/core/base.hpp>
#include <occa/tools/env.hpp>
#include <occa/io.hpp>
#include <occa/mode/serial/kernel.hpp>
#include <occa/lang/mode/serial.hpp>

namespace occa {
  namespace serial {
    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   const occa::properties &properties_) :
      occa::modeKernel_t(modeDevice_, name_, sourceFilename_, properties_) {
      dlHandle = NULL;
      function = NULL;
    }

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
      return dim(-1,-1,-1);
    }

    dim kernel::maxInnerDims() const {
      return dim(-1,-1,-1);
    }

    void kernel::run() const {
      const int totalArgCount = kernelArg::argumentCount(arguments);
      if (!totalArgCount) {
        vArgs.resize(1);
      } else if ((int) vArgs.size() < totalArgCount) {
        vArgs.resize(totalArgCount);
      }

      const int kArgCount = (int) arguments.size();

      int argc = 0;
      for (int i = 0; i < kArgCount; ++i) {
        const kArgVector &iArgs = arguments[i].args;
        const int argCount = (int) iArgs.size();
        if (!argCount) {
          continue;
        }
        for (int ai = 0; ai < argCount; ++ai) {
          vArgs[argc++] = iArgs[ai].ptr();
        }
      }

      sys::runFunction(function, argc, &(vArgs[0]));
    }
  }
}
