#include <occa/defines.hpp>

#if OCCA_HIP_ENABLED

#include <occa/mode/hip/kernel.hpp>
#include <occa/mode/hip/device.hpp>
#include <occa/mode/hip/utils.hpp>
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

    int kernel::maxDims() const {
      return 3;
    }

    dim kernel::maxOuterDims() const {
      return dim(-1, -1, -1);
    }

    dim kernel::maxInnerDims() const {
      static dim innerDims(0);
      if (innerDims.x == 0) {
        int maxSize;
        int deviceID = properties["device_id"];
        hipDeviceProp_t props;
        OCCA_HIP_ERROR("Getting device properties",
                       hipGetDeviceProperties(&props, deviceID ));
        maxSize = props.maxThreadsPerBlock;

        innerDims.x = maxSize;
      }
      return innerDims;
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
      for (int i = 0; i < kArgCount; ++i) {
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
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &(vArgs[0]),
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
        HIP_LAUNCH_PARAM_END
      };

      OCCA_HIP_ERROR("Launching Kernel",
                     hipModuleLaunchKernel(hipFunction,
                                           outerDims.x, outerDims.y, outerDims.z,
                                           innerDims.x, innerDims.y, innerDims.z,
                                           0, *((hipStream_t*) modeDevice->currentStream),
                                           NULL, (void**) &config));
    }
  }
}

#endif
