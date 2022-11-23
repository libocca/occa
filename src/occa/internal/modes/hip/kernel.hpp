#ifndef OCCA_INTERNAL_MODES_HIP_KERNEL_HEADER
#define OCCA_INTERNAL_MODES_HIP_KERNEL_HEADER

#include <occa/internal/core/launchedKernel.hpp>
#include <occa/internal/modes/hip/polyfill.hpp>

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
             hipModule_t   hipModule_,
             const occa::json &properties_);

      kernel(modeDevice_t *modeDevice_,
             const std::string &name_,
             const std::string &sourceFilename_,
             hipFunction_t hipFunction_,
             const occa::json &properties_);

      kernel(modeDevice_t *modeDevice_,
             const std::string &name_,
             const std::string &sourceFilename_,
             hipModule_t   hipModule_,
             hipFunction_t hipFunction_,
             const occa::json &properties_);

      virtual ~kernel();

      hipStream_t& getHipStream() const;

      int maxDims() const override;
      dim maxOuterDims() const override;
      dim maxInnerDims() const override;

      void deviceRun() const override;
    };
  }
}

#endif
