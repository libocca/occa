#include <occa/core/base.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/io.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/internal/modes/opencl/kernel.hpp>
#include <occa/internal/modes/opencl/device.hpp>
#include <occa/internal/modes/opencl/utils.hpp>

namespace occa {
  namespace opencl {
    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   const occa::json &properties_) :
      occa::launchedModeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
      clDevice(NULL),
      clKernel(NULL) {}

    kernel::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   const std::string &sourceFilename_,
                   cl_device_id clDevice_,
                   cl_kernel clKernel_,
                   const occa::json &properties_) :
      occa::launchedModeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
      clDevice(clDevice_),
      clKernel(clKernel_) {}

    kernel::~kernel() {
      if (clKernel) {
        OCCA_OPENCL_ERROR("Kernel [" + name + "]: Free",
                          clReleaseKernel(clKernel));
        clKernel = NULL;
      }
    }

    cl_command_queue& kernel::getCommandQueue() const {
      return ((device*) modeDevice)->getCommandQueue();
    }

    int kernel::maxDims() const {
      // TODO 1.1: This should be in the device, not the kernel
      static cl_uint dims_ = 0;
      if (dims_ == 0) {
        size_t bytes;
        OCCA_OPENCL_ERROR("Kernel: Max Dims",
                          clGetDeviceInfo(clDevice,
                                          CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                          0, NULL, &bytes));
        OCCA_OPENCL_ERROR("Kernel: Max Dims",
                          clGetDeviceInfo(clDevice,
                                          CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                          bytes, &dims_, NULL));
      }
      return (int) dims_;
    }

    dim kernel::maxOuterDims() const {
      // TODO 1.1: This should be in the device, not the kernel
      static occa::dim maxOuterDims_(0);
      if (maxOuterDims_.x == 0) {
        int dims_ = maxDims();
        size_t *od = new size_t[dims_];
        size_t bytes;
        OCCA_OPENCL_ERROR("Kernel: Max Outer Dims",
                          clGetDeviceInfo(clDevice,
                                          CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                          0, NULL, &bytes));
        OCCA_OPENCL_ERROR("Kernel: Max Outer Dims",
                          clGetDeviceInfo(clDevice,
                                          CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                          bytes, &od, NULL));
        for (int i = 0; i < dims_; ++i) {
          maxOuterDims_[i] = od[i];
        }
        delete [] od;
      }
      return maxOuterDims_;
    }

    dim kernel::maxInnerDims() const {
      // TODO 1.1: This should be in the device, not the kernel
      static occa::dim maxInnerDims_(0);
      if (maxInnerDims_.x == 0) {
        size_t dims_ = 0;
        size_t bytes = 0;
        OCCA_OPENCL_ERROR("Kernel: Max Inner Dims",
                          clGetKernelWorkGroupInfo(clKernel,
                                                   clDevice,
                                                   CL_KERNEL_WORK_GROUP_SIZE,
                                                   0, NULL, &bytes));
        OCCA_OPENCL_ERROR("Kernel: Max Inner Dims",
                          clGetKernelWorkGroupInfo(clKernel,
                                                   clDevice,
                                                   CL_KERNEL_WORK_GROUP_SIZE,
                                                   bytes, &dims_, NULL));
        maxInnerDims_.x = dims_;
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
      const int args = (int) arguments.size();
      for (int i = 0; i < args; ++i) {
        const kernelArgData &arg = arguments[i];
        OCCA_OPENCL_ERROR("Kernel [" + name + "]"
                          << ": Setting Kernel Argument [" << (i + 1) << "]",
                          clSetKernelArg(clKernel, i, arg.size(), arg.ptr()));
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
