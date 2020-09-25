#include <occa/core/base.hpp>
#include <occa/tools/env.hpp>
#include <occa/io.hpp>
#include <occa/tools/sys.hpp>
#include <occa/modes/dpcpp/kernel.hpp>
#include <occa/modes/dpcpp/device.hpp>
#include <occa/modes/dpcpp/utils.hpp>

namespace occa {
  namespace dpcpp {
    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   const occa::properties &properties_) :
      occa::launchedModeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
      dpcppDevice(NULL),
      dpcppKernel(NULL) {}

    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   ::sycl::device dpcppDevice_,
                   ::sycl::kernel dpcppKernel_,
                   const occa::properties &properties_) :
      occa::launchedModeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
      dpcppDevice(dpcppDevice_),
      dpcppKernel(dpcppKernel_) {}

    kernel::~kernel() {
      if (dpcppKernel) {
        clKernel = NULL;
      }
    }

    ::sycl::queue& kernel::getCommandQueue() const {
      return ((device*) modeDevice)->getCommandQueue();
    }

    int kernel::maxDims() const {
      static cl_uint dims_ = 0;
      dims_ = dpcppDevice.get_info<sycl::info::device::max_work_item_dimensions>();
      return (int) dims_;
    }

    dim kernel::maxOuterDims() const {
      // TODO 1.1: This should be in the device, not the kernel
      static occa::dim maxOuterDims_(0);
      if (maxOuterDims_.x == 0) {
        int dims_ = maxDims();
        id<3> od=dpcppDevice.get_info<sycl::info::device::max_work_item_sizes>();
        for (int i = 0; i < dims_; ++i) {
          maxOuterDims_[i] = (size_t)od[i];
        }
      }
      return maxOuterDims_;
    }

    dim kernel::maxInnerDims() const {
      // TODO 1.1: This should be in the device, not the kernel
      static occa::dim maxInnerDims_(0);
      if (maxInnerDims_.x == 0) {
        dims_ = dpcppDevice.get_info<sycl::info::device::max_work_group_size>();
        maxInnerDims_.x = dims_;
        maxInnerDims_.y = dims_;
        maxInnerDims_.z = dims_;
      }
      return maxInnerDims_;
    }

    void kernel::deviceRun() const {
      // Setup kernel dimensions
      occa::dim fullDims = (outerDims * innerDims);

      size_t fullDims_[3] = {
        fullDims.x, fullDims.y, fullDims.z
      };
      size_t innerDims_[3] = {
        innerDims.x, innerDims.y, innerDims.z
      };

      // Set arguments
      const int args = :(int) arguments.size();
      for (int i = 0; i < args; ++i) {
        const kernelArgData &arg = arguments[i];
        OCCA_OPENCL_ERROR("Kernel [" + name + "]"
                          << ": Setting Kernel Argument [" << (i + 1) << "]",
                          clSetKernelArg(clKernel, i, arg.size, arg.ptr()));
      }

      OCCA_OPENCL_ERROR("Kernel [" + name + "]"
                        << " : Kernel Run",
                        clEnqueueNDRangeKernel(getCommandQueue(),
                                               clKernel,
                                               (cl_int) fullDims.dims,
                                               NULL,
                                               (size_t*) &fullDims_,
                                               (size_t*) &innerDims_,
                                               0, NULL, NULL));
    }
  }
}
