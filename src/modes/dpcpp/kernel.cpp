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
      occa::modeKernel_t(modeDevice_, name_, sourceFilename_, properties_),
      dlHandle(NULL),
      function(NULL),
      isLauncherKernel(true) {
      properties["compiler"] = "dpcpp";
      properties["compiler_linker_flags"] = "-shared -fPIC";

      }



    kernel::~kernel() {
    }

    ::sycl::queue* kernel::getCommandQueue() const {
      return ((device*) modeDevice)->getCommandQueue();
    }

    int kernel::maxDims() const {
      static cl_uint dims_ = 0;
      dims_ = getCommandQueue()->get_device().get_info<::sycl::info::device::max_work_item_dimensions>();
      return (int) dims_;
    }

    const lang::kernelMetadata_t& kernel::getMetadata() const {
      return metadata;
    }

    dim kernel::maxOuterDims() const {
      // TODO 1.1: This should be in the device, not the kernel
      static occa::dim maxOuterDims_(0);
      if (maxOuterDims_.x == 0) {
        int dims_ = maxDims();
        ::sycl::id<3> od=getCommandQueue()->get_device().get_info<::sycl::info::device::max_work_item_sizes>();
        for (int i = 0; i < dims_; ++i) {
          maxOuterDims_[i] = (size_t)od[i];
        }
      }
      return maxOuterDims_;
    }

    dim kernel::maxInnerDims() const {
      // TODO 1.1: This should be in the device, not the kernel
      static cl_uint dims_ = 0;
      static occa::dim maxInnerDims_(0);
      if (maxInnerDims_.x == 0) {
        dims_ = getCommandQueue()->get_device().get_info<::sycl::info::device::max_work_group_size>();
        maxInnerDims_.x = dims_;
        maxInnerDims_.y = dims_;
        maxInnerDims_.z = dims_;
      }
      return maxInnerDims_;
    }

    void kernel::run() const {
      // Setup kernel dimensions
      occa::dim fullDims = (outerDims * innerDims);
      ::sycl::queue *q = getCommandQueue();
      int odimx = (innerDims.x % outerDims.x == 0)? innerDims.x : outerDims.x * ((innerDims.x / outerDims.x)+1);
      int odimy = (innerDims.y % outerDims.y == 0)? innerDims.y : outerDims.y * ((innerDims.y / outerDims.y)+1);
      int odimz = (innerDims.z % outerDims.z == 0)? innerDims.z : outerDims.z * ((innerDims.z / outerDims.z)+1);
      auto global_range = ::sycl::range<3>(odimx, odimy, odimz);
      auto local_range  = ::sycl::range<3>(outerDims.x, outerDims.y, outerDims.z);
      std::vector<void*> args;
      ::sycl::nd_range<3> ndrange = ::sycl::nd_range<3>(global_range, local_range);
      args.push_back(q);
      args.push_back(&ndrange);
      for(int i=0; i<arguments.size(); i++){
	args.push_back(arguments[i].ptr());
      }
      sys::runFunction(function, args.size(), &(args[0]));
    }
  }
}
