#ifndef OCCA_MODES_HIP_KERNEL_HEADER
#define OCCA_MODES_HIP_KERNEL_HEADER

#include <occa/core/launchedKernel.hpp>
#include <occa/modes/hip/polyfill.hpp>

namespace occa {
  namespace hip {
    class device;

    class kernel : public occa::launchedModeKernel_t {
      friend class device;

    private:
      hipModule_t hipModule;
      hipFunction_t hipFunction;

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

      hipStream_t& getHipStream() const;

      int maxDims() const;
      dim maxOuterDims() const;
      dim maxInnerDims() const;

      void deviceRun() const;
    };
  }
}

#endif
