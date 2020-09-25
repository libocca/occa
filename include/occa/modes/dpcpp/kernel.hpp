#ifndef OCCA_MODES_DPCPP_KERNEL_HEADER
#define OCCA_MODES_DPCPP_KERNEL_HEADER

#include <occa/core/launchedKernel.hpp>
#include <occa/modes/dpcpp/polyfill.hpp>
#include <occa/modes/dpcpp/utils.hpp>
#include <CL/sycl.hpp>

namespace occa {
  namespace dpcpp {
    class device;

    template <class T> class kernel : public occa::launchedModeKernel_t {
      friend class device;
      friend cl_kernel getCLKernel(occa::kernel kernel);

    private:
<<<<<<< HEAD
      ::sycl::device dpcppDevice;
      ::sycl::kernel dpcppKernel;
     
=======
      ::sycl::device *dpcppDevice;
      T* dpcppKernel; 
>>>>>>> parent of 8e2aacde... Revert "It compiles now :) but won't work before we find a way to enqueue arguments"

    public:
      kernel(modeDevice_t *modeDevice_,
             const std::string &name_,
<<<<<<< HEAD
             const std::string &sourceFilename_,
             const occa::properties &properties_);

      kernel(modeDevice_t *modeDevice_,
             const std::string &name_,
             const std::string &sourceFilename_,
             ::sycl::device dpcppDevice_,
             ::sycl::kernel dpcppKernel_,
             const occa::properties &properties_);
=======
             ::sycl::device* dpcppDevice_,
             T* lambda_,
             const occa::properties &properties_);

      kernel(modeDevice_t *modeDevice_,
	     const std::string &name_, const occa::properties &properties_, T* lambda_);
>>>>>>> parent of 8e2aacde... Revert "It compiles now :) but won't work before we find a way to enqueue arguments"

      ~kernel();

      queue& getCommandQueue() const;

      int maxDims() const;
      dim maxOuterDims() const;
      dim maxInnerDims() const;

      void deviceRun() const;
    };
  }
}

#endif
