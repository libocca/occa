#include <occa/defines.hpp>

#ifndef OCCA_MODES_CUDA_KERNEL_HEADER
#define OCCA_MODES_CUDA_KERNEL_HEADER

#include <occa/core/launchedKernel.hpp>
#include <occa/modes/cuda/polyfill.hpp>

namespace occa {
  namespace cuda {
    class device;

    class kernel : public occa::launchedModeKernel_t {
      friend class device;

    private:
      CUmodule cuModule;
      CUfunction cuFunction;

      mutable std::vector<void*> vArgs;

    public:
      kernel(modeDevice_t *modeDevice_,
             const std::string &name_,
             const std::string &sourceFilename_,
             const occa::properties &properties_);

      kernel(modeDevice_t *modeDevice_,
             const std::string &name_,
             const std::string &sourceFilename_,
             CUmodule cuModule_,
             CUfunction cuFunction_,
             const occa::properties &properties_);

      ~kernel();

      CUstream& getCuStream() const;

      int maxDims() const;
      dim maxOuterDims() const;
      dim maxInnerDims() const;

      void deviceRun() const;
    };
  }
}

#endif
