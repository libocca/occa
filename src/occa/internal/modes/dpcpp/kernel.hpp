#ifndef OCCA_MODES_DPCPP_KERNEL_HEADER
#define OCCA_MODES_DPCPP_KERNEL_HEADER

#include <occa/internal/core/launchedKernel.hpp>
#include <occa/internal/modes/dpcpp/polyfill.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa
{
  namespace dpcpp
  {
    class device;

    class kernel : public occa::launchedModeKernel_t
    {
      friend class device;

//@todo: Check public/private/protected here
    public:
      void *dlHandle{nullptr};
      functionPtr_t function{nullptr};

    public:
      kernel(modeDevice_t *modeDevice_,
             const std::string &name_,
             const std::string &sourceFilename_,
             void* dlHandle_,
             const occa::json &properties_);

      kernel(modeDevice_t *modeDevice_,
             const std::string &name_,
             const std::string &sourceFilename_,
             functionPtr_t function_,
             const occa::json &properties_);

      kernel(modeDevice_t *modeDevice_,
             const std::string &name_,
             const std::string &sourceFilename_,
             void* dlHandle_,
             functionPtr_t function_,
             const occa::json &properties_);

      virtual ~kernel();

      int maxDims() const override;
      dim maxOuterDims() const override;
      dim maxInnerDims() const override;
      udim_t maxInnerSize() const;

      void deviceRun() const override;
    };
  } // namespace dpcpp
} // namespace occa

#endif
