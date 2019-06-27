#ifndef OCCA_MODES_OPENCL_KERNEL_HEADER
#define OCCA_MODES_OPENCL_KERNEL_HEADER

#include <occa/core/launchedKernel.hpp>
#include <occa/modes/opencl/polyfill.hpp>
#include <occa/modes/opencl/utils.hpp>

namespace occa {
  namespace opencl {
    class device;

    class kernel : public occa::launchedModeKernel_t {
      friend class device;
      friend cl_kernel getCLKernel(occa::kernel kernel);

    private:
      cl_device_id clDevice;
      cl_kernel clKernel;

    public:
      kernel(modeDevice_t *modeDevice_,
             const std::string &name_,
             const std::string &sourceFilename_,
             const occa::properties &properties_);

      kernel(modeDevice_t *modeDevice_,
             const std::string &name_,
             const std::string &sourceFilename_,
             cl_device_id clDevice_,
             cl_kernel clKernel_,
             const occa::properties &properties_);

      ~kernel();

      cl_command_queue& getCommandQueue() const;

      int maxDims() const;
      dim maxOuterDims() const;
      dim maxInnerDims() const;

      void deviceRun() const;
    };
  }
}

#endif
