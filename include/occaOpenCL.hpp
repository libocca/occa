#if OCCA_OPENCL_ENABLED
#  ifndef OCCA_OPENCL_HEADER
#  define OCCA_OPENCL_HEADER

#include "occaBase.hpp"
#include "occaPods.hpp"

#include "occaKernelDefines.hpp"

#include "occaDefines.hpp"

#if   OCCA_OS == LINUX_OS
#  include <CL/cl.h>
#  include <CL/cl_gl.h>
#elif OCCA_OS == OSX_OS
#  include <OpenCL/OpenCl.h>
#endif

namespace occa {
  //---[ Data Structs ]---------------
  struct OpenCLKernelData_t {
    int platform, device;

    cl_platform_id platformID;
    cl_device_id deviceID;
    cl_context   context;
    cl_program   program;
    cl_kernel    kernel;
  };

  struct OpenCLDeviceData_t {
    int platform, device;

    cl_platform_id platformID;
    cl_device_id   deviceID;
    cl_context     context;
  };
  //==================================


  //---[ Kernel ]---------------------
  template <>
  kernel_t<OpenCL>::kernel_t();

  template <>
  kernel_t<OpenCL>::kernel_t(const kernel_t &k);

  template <>
  kernel_t<OpenCL>& kernel_t<OpenCL>::operator = (const kernel_t<OpenCL> &k);

  template <>
  kernel_t<OpenCL>::kernel_t(const kernel_t<OpenCL> &k);

  template <>
  kernel_t<OpenCL>* kernel_t<OpenCL>::buildFromSource(const std::string &filename,
                                                      const std::string &functionName_,
                                                      const kernelInfo &info_);

  template <>
  kernel_t<OpenCL>* kernel_t<OpenCL>::buildFromBinary(const std::string &filename,
                                                      const std::string &functionName_);

  template <>
  int kernel_t<OpenCL>::preferredDimSize();

  template <>
  double kernel_t<OpenCL>::timeTaken();

  template <>
  void kernel_t<OpenCL>::free();
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<OpenCL>::memory_t();

  template <>
  memory_t<OpenCL>::memory_t(const memory_t &m);

  template <>
  memory_t<OpenCL>& memory_t<OpenCL>::operator = (const memory_t &m);

  template <>
  void memory_t<OpenCL>::copyFrom(const void *source,
                                  const size_t bytes,
                                  const size_t offset);

  template <>
  void memory_t<OpenCL>::copyFrom(const memory_v *source,
                                  const size_t bytes,
                                  const size_t offset);

  template <>
  void memory_t<OpenCL>::copyTo(void *dest,
                                const size_t bytes,
                                const size_t offset);

  template <>
  void memory_t<OpenCL>::copyTo(memory_v *dest,
                                const size_t bytes,
                                const size_t offset);

  template <>
  void memory_t<OpenCL>::asyncCopyFrom(const void *source,
                                       const size_t bytes,
                                       const size_t offset);

  template <>
  void memory_t<OpenCL>::asyncCopyFrom(const memory_v *source,
                                       const size_t bytes,
                                       const size_t offset);

  template <>
  void memory_t<OpenCL>::asyncCopyTo(void *dest,
                                     const size_t bytes,
                                     const size_t offset);

  template <>
  void memory_t<OpenCL>::asyncCopyTo(memory_v *dest,
                                     const size_t bytes,
                                     const size_t offset);

  template <>
  void memory_t<OpenCL>::free();
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<OpenCL>::device_t();

  template <>
  device_t<OpenCL>::device_t(const device_t<OpenCL> &k);

  template <>
  device_t<OpenCL>::device_t(const int platform, const int device);

  template <>
  device_t<OpenCL>& device_t<OpenCL>::operator = (const device_t<OpenCL> &k);

  template <>
  void device_t<OpenCL>::setup(const int platform, const int device);

  template <>
  void device_t<OpenCL>::flush();

  template <>
  void device_t<OpenCL>::finish();

  template <>
  stream device_t<OpenCL>::genStream();

  template <>
  void device_t<OpenCL>::freeStream(stream s);

  template <>
  kernel_v* device_t<OpenCL>::buildKernelFromSource(const std::string &filename,
                                                    const std::string &functionName_,
                                                    const kernelInfo &info_);

  template <>
  kernel_v* device_t<OpenCL>::buildKernelFromBinary(const std::string &filename,
                                                    const std::string &functionName_);

  template <>
  memory_v* device_t<OpenCL>::malloc(const size_t bytes,
                                     void *source);

  template <>
  int device_t<OpenCL>::simdWidth();
  //==================================


  //---[ Error Handling ]-------------
  std::string openclError(int e);
  //==================================
};

#  endif
#endif
