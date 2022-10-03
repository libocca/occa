#include <occa/internal/modes/dpcpp/kernel.hpp>
#include <occa/internal/modes/dpcpp/device.hpp>
#include <occa/internal/modes/dpcpp/utils.hpp>
#include <occa/internal/modes/dpcpp/stream.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/io.hpp>
#include <occa/core/base.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa
{
  namespace dpcpp
  {
    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   void *dlHandle_,
                   const occa::json &properties_)
        : occa::launchedModeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
          dlHandle{dlHandle_},
          function{nullptr}
    {
    }

    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   functionPtr_t function_,
                   const occa::json &properties_)
        : occa::launchedModeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
          dlHandle(nullptr),
          function(function_)
    {
    }

    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   void *dlHandle_,
                   functionPtr_t function_,
                   const occa::json &properties_)
        : occa::launchedModeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
          dlHandle(dlHandle_),
          function(function_)
    {
    }

    kernel::~kernel()
    {
      if (dlHandle)
      {
        sys::dlclose(dlHandle);
        dlHandle = nullptr;
      }
      function = nullptr;
    }

    void kernel::deviceRun() const
    {
      // Setup kernel dimensions
      occa::dim fullDims = (outerDims * innerDims);
      ::sycl::range<3> global_range{fullDims.z, fullDims.y, fullDims.x};
      ::sycl::range<3> local_range{innerDims.z, innerDims.y, innerDims.x};
      ::sycl::nd_range<3> ndrange{global_range, local_range};

      std::vector<void *> args;

      auto& q{getDpcppStream(modeDevice->currentStream).commandQueue};

      args.push_back(&q);
      args.push_back(&ndrange);
      for (size_t i = 0; i < arguments.size(); ++i)
      {
        args.push_back(arguments[i].ptr());
      }

      sys::runFunction(function, args.size(), &(args[0]));
    }

    int kernel::maxDims() const
    {
      return occa::dpcpp::max_dimensions;
    }

    dim kernel::maxOuterDims() const
    {
      return getDpcppDevice(modeDevice).maxOuterDims();
    }

    dim kernel::maxInnerDims() const
    {
      return getDpcppDevice(modeDevice).maxInnerDims();
    }

    udim_t kernel::maxInnerSize() const
    {
      return getDpcppDevice(modeDevice).maxInnerSize();
    }
  } // namespace dpcpp
} // namespace occa
