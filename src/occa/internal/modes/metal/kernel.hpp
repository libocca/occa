#ifndef OCCA_INTERNAL_MODES_METAL_KERNEL_HEADER
#define OCCA_INTERNAL_MODES_METAL_KERNEL_HEADER

#include <occa/internal/core/launchedKernel.hpp>
#include <occa/internal/api/metal.hpp>

namespace occa {
  namespace metal {
    class device;

    class kernel : public occa::launchedModeKernel_t {
      friend class device;

    private:
      api::metal::device_t metalDevice;
      mutable api::metal::function_t metalFunction;

    public:
      kernel(modeDevice_t *modeDevice_,
             const std::string &name_,
             const std::string &sourceFilename_,
             const occa::json &properties_);

      kernel(modeDevice_t *modeDevice_,
             const std::string &name_,
             const std::string &sourceFilename_,
             api::metal::device_t metalDevice_,
             api::metal::function_t metalFunction_,
             const occa::json &properties_);

      virtual ~kernel();

      int maxDims() const override;
      dim maxOuterDims() const override;
      dim maxInnerDims() const override;

      void deviceRun() const override;
    };
  }
}

#endif
