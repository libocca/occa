#include <occa/defines.hpp>

#ifndef OCCA_INTERNAL_MODES_CUDA_KERNEL_HEADER
#define OCCA_INTERNAL_MODES_CUDA_KERNEL_HEADER

#include <occa/internal/core/launchedKernel.hpp>
#include <occa/internal/modes/cuda/polyfill.hpp>

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
             const occa::json &properties_);

      kernel(modeDevice_t *modeDevice_,
             const std::string &name_,
             const std::string &sourceFilename_,
             CUmodule cuModule_,
             CUfunction cuFunction_,
             const occa::json &properties_);

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
