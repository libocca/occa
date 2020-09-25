#include <occa/core/base.hpp>
#include <occa/tools/env.hpp>
#include <occa/io.hpp>
#include <occa/tools/sys.hpp>
#include <occa/modes/dpcpp/kernel.hpp>
#include <occa/modes/dpcpp/device.hpp>
#include <occa/modes/dpcpp/utils.hpp>

namespace occa {
  namespace dpcpp {
    template <class T> kernel<T>::kernel(modeDevice_t *modeDevice_,
             const std::string &name_, const occa::properties &properties_, T t):
      occa::launchedModeKernel_t(modeDevice_, name_, "", properties_),
      dpcppDevice(NULL){}


    template <class T> kernel<T>::kernel(modeDevice_t *modeDevice_,
                   const std::string &name_,
                   ::sycl::device* dpcppDevice_,
                   const occa::properties &properties_) :
      occa::launchedModeKernel_t(modeDevice_, name_, "", properties_),
      dpcppDevice(dpcppDevice_){}

    template <class T> kernel<T>::~kernel() {
      if (dpcppKernel) {
        dpcppKernel = NULL;
      }
    }

    template <class T> ::sycl::queue* kernel<T>::getCommandQueue() const {
      return ((device*) modeDevice)->getCommandQueue();
    }

    template <class T> int kernel<T>::maxDims() const {
      static cl_uint dims_ = 0;
      dims_ = dpcppDevice->get_info<sycl::info::device::max_work_item_dimensions>();
      return (int) dims_;
    }

    template <class T> dim kernel<T>::maxOuterDims() const {
      // TODO 1.1: This should be in the device, not the kernel
      static occa::dim maxOuterDims_(0);
      if (maxOuterDims_.x == 0) {
        int dims_ = maxDims();
        ::sycl::id<3> od=dpcppDevice->get_info<sycl::info::device::max_work_item_sizes>();
        for (int i = 0; i < dims_; ++i) {
          maxOuterDims_[i] = (size_t)od[i];
        }
      }
      return maxOuterDims_;
    }

    template <class T> dim kernel<T>::maxInnerDims() const {
      // TODO 1.1: This should be in the device, not the kernel
      static occa::dim maxInnerDims_(0);
      static cl_uint dims_ = 0;
      if (maxInnerDims_.x == 0) {
        dims_ = dpcppDevice->get_info<sycl::info::device::max_work_group_size>();
        maxInnerDims_.x = dims_;
        maxInnerDims_.y = dims_;
        maxInnerDims_.z = dims_;
      }
      return maxInnerDims_;
    }

    template <class T> void kernel<T>::deviceRun() const {
      // Setup kernel dimensions
      occa::dim fullDims = (outerDims * innerDims);
      ::sycl::queue *q = getCommandQueue();
      auto global_range = ::sycl::range<3>(outerDims.x, outerDims.y, outerDims.z);
      auto local_range  = ::sycl::range<3>(innerDims.x, innerDims.y, innerDims.z);

      const int args = (int) arguments.size();
      T* func = new T();
      for (int i = 0; i < args; ++i) {
        const kernelArgData &arg = arguments[i];
	void **add = (void**) func->get_member_adress(i);
	*add = arg.ptr();
      }

      q->submit([&](::sycl::handler &h){

	h.parallel_for(::sycl::nd_range<3>{global_range, local_range},
					  *func);
      });
      free(func);
/*      size_t fullDims_[3] = {
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
 */   }
  }
}
