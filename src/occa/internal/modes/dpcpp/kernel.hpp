#ifndef OCCA_MODES_DPCPP_KERNEL_HEADER
#define OCCA_MODES_DPCPP_KERNEL_HEADER

#include <occa/internal/core/launchedKernel.hpp>
#include <occa/internal/modes/dpcpp/polyfill.hpp>
#include <occa/internal/modes/dpcpp/utils.hpp>

namespace occa
{
  namespace dpcpp
  {
    class device;

    class kernel : public occa::launchedModeKernel_t
    {
      friend class device;

    private:
      ::sycl::device *dpcppDevice;
      functionPtr_t function;
      void *dlHandle;

    public:
      kernel(modeDevice_t *modeDevice_,
             const std::string &name_,
             const std::string &sourceFilename_,
             const occa::json &properties_);

      kernel(modeDevice_t *modeDevice_,
             const std::string &name_,
             const std::string &sourceFilename_,
             ::sycl::device *dpcppDevice_,
             functionPtr_t function_,
             const occa::json &properties_);

      ~kernel();

      ::sycl::queue *getCommandQueue() const;
      const lang::kernelMetadata_t &getMetadata() const;

      int maxDims() const;
      dim maxOuterDims() const;
      dim maxInnerDims() const;

      void deviceRun() const;
    };
  } // namespace dpcpp
} // namespace occa

#endif
