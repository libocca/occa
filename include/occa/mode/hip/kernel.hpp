#include <occa/defines.hpp>

#if OCCA_HIP_ENABLED
#  ifndef OCCA_MODES_HIP_KERNEL_HEADER
#  define OCCA_MODES_HIP_KERNEL_HEADER

#include <hip/hip_runtime_api.h>

#include <occa/core/kernel.hpp>

namespace occa {
  namespace hip {
    class device;

    class kernel : public occa::modeKernel_t {
      friend class device;

    private:
      hipModule_t hipModule;
      hipFunction_t hipFunction;

      occa::modeKernel_t *launcherKernel;
      std::vector<kernel*> hipKernels;
      mutable std::vector<void*> vArgs;

    public:
      kernel(modeDevice_t *modeDevice_,
             const std::string &name_,
             const std::string &sourceFilename_,
             const occa::properties &properties_);

      kernel(modeDevice_t *modeDevice_,
             const std::string &name_,
             const std::string &sourceFilename_,
             hipModule_t   hipModule_,
             hipFunction_t hipFunction_,
             const occa::properties &properties_);

      ~kernel();

      int maxDims() const;
      dim maxOuterDims() const;
      dim maxInnerDims() const;

      void deviceRun() const;
    };
  }
}

#  endif
#endif
