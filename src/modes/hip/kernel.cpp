#include <occa/modes/hip/kernel.hpp>
#include <occa/modes/hip/device.hpp>
#include <occa/modes/hip/utils.hpp>
#include <occa/tools/env.hpp>
#include <occa/io.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace hip {
    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   const occa::properties &properties_) :
      occa::launchedModeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
      hipModule(NULL),
      hipFunction(NULL) {}

    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   hipModule_t hipModule_,
                   hipFunction_t hipFunction_,
                   const occa::properties &properties_) :
      occa::launchedModeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
      hipModule(hipModule_),
      hipFunction(hipFunction_) {}

    kernel::~kernel() {
      if (hipModule) {
        OCCA_HIP_ERROR("Kernel (" + name + ") : Unloading Module",
                       hipModuleUnload(hipModule));
        hipModule = NULL;
      }
    }

    hipStream_t& kernel::getHipStream() const {
      return ((device*) modeDevice)->getHipStream();
    }

    int kernel::maxDims() const {
      return 3;
    }

    dim kernel::maxOuterDims() const {
      return dim(-1, -1, -1);
    }

    dim kernel::maxInnerDims() const {
      static dim _maxInnerDims(0);
      if (_maxInnerDims.x == 0) {
        int maxSize;
        int deviceID = properties["device_id"];
        hipDeviceProp_t props;
        OCCA_HIP_ERROR("Getting device properties",
                       hipGetDeviceProperties(&props, deviceID));
        maxSize = props.maxThreadsPerBlock;

        _maxInnerDims.x = maxSize;
      }
      return _maxInnerDims;
    }

    void kernel::deviceRun() const {
      const int args = (int) arguments.size();
      if (!args) {
        vArgs.resize(1);
      } else if ((int) vArgs.size() < args) {
        vArgs.resize(args);
      }

      // HIP expects kernel arguments to be byte-aligned so we add padding to arguments
      char *dataPtr = (char*) &(vArgs[0]);
      int padding = 0;
      for (int i = 0; i < args; ++i) {
        const kernelArgData &arg = arguments[i];

        size_t bytes;
        if ((padding + arg.size) <= sizeof(void*)) {
          bytes = arg.size;
          padding = sizeof(void*) - padding - arg.size;
        } else {
          bytes = sizeof(void*);
          dataPtr += padding;
          padding = 0;
        }

        ::memcpy(dataPtr,
                 &(arg.data.int64_),
                 bytes);
        dataPtr += bytes;
      }

      size_t size = vArgs.size() * sizeof(vArgs[0]);
      void* config[] = {
        (void*) HIP_LAUNCH_PARAM_BUFFER_POINTER, &(vArgs[0]),
        (void*) HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
        (void*) HIP_LAUNCH_PARAM_END
      };

      OCCA_HIP_ERROR("Launching Kernel",
                     hipModuleLaunchKernel(hipFunction,
                                           outerDims.x, outerDims.y, outerDims.z,
                                           innerDims.x, innerDims.y, innerDims.z,
                                           0, getHipStream(),
                                           NULL, (void**) &config));
    }
  }
}
