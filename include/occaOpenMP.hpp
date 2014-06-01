#ifndef OCCA_OPENMP_HEADER
#define OCCA_OPENMP_HEADER

#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <dlfcn.h>
#include <fcntl.h>

#include "occaBase.hpp"
#include "occaPods.hpp"

#include "occaKernelDefines.hpp"

namespace occa {
  //---[ Data Structs ]---------------
  struct OpenMPKernelData_t {
    void *handle;
  };

  class occaArgs_t {
  public:
    int data[12];

    inline occaArgs_t(dim inner, dim outer){
      data[0] = outer.z;
      data[2] = outer.y;
      data[4] = outer.x;

      data[6]  = inner.z;
      data[8]  = inner.y;
      data[10] = inner.x;
    }
  };
  //==================================


  //---[ Kernel ]---------------------
  OCCA_OPENMP_FUNCTION_POINTER_TYPEDEFS;

  template <>
  kernel_t<OpenMP>::kernel_t();

  template <>
  kernel_t<OpenMP>::kernel_t(const kernel_t &k);

  template <>
  kernel_t<OpenMP>& kernel_t<OpenMP>::operator = (const kernel_t<OpenMP> &k);

  template <>
  kernel_t<OpenMP>::kernel_t(const kernel_t<OpenMP> &k);

  template <>
  kernel_t<OpenMP>* kernel_t<OpenMP>::buildFromSource(const std::string &filename,
                                                      const std::string &functionName_,
                                                      const kernelInfo &info_);

  template <>
  kernel_t<OpenMP>* kernel_t<OpenMP>::buildFromBinary(const std::string &filename,
                                                      const std::string &functionName_);

  template <>
  int kernel_t<OpenMP>::preferredDimSize();

  template <>
  double kernel_t<OpenMP>::timeTaken();

  template <>
  void kernel_t<OpenMP>::free();
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<OpenMP>::memory_t();

  template <>
  memory_t<OpenMP>::memory_t(const memory_t &m);

  template <>
  memory_t<OpenMP>& memory_t<OpenMP>::operator = (const memory_t &m);

  template <>
  void memory_t<OpenMP>::copyFrom(const void *source,
                                  const size_t bytes,
                                  const size_t offset);

  template <>
  void memory_t<OpenMP>::copyFrom(const memory_v *source,
                                  const size_t bytes,
                                  const size_t offset);

  template <>
  void memory_t<OpenMP>::copyTo(void *dest,
                                const size_t bytes,
                                const size_t offset);

  template <>
  void memory_t<OpenMP>::copyTo(memory_v *dest,
                                const size_t bytes,
                                const size_t offset);

  template <>
  void memory_t<OpenMP>::asyncCopyFrom(const void *source,
                                       const size_t bytes,
                                       const size_t offset);

  template <>
  void memory_t<OpenMP>::asyncCopyFrom(const memory_v *source,
                                       const size_t bytes,
                                       const size_t offset);

  template <>
  void memory_t<OpenMP>::asyncCopyTo(void *dest,
                                     const size_t bytes,
                                     const size_t offset);

  template <>
  void memory_t<OpenMP>::asyncCopyTo(memory_v *dest,
                                     const size_t bytes,
                                     const size_t offset);

  template <>
  void memory_t<OpenMP>::free();
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<OpenMP>::device_t();

  template <>
  device_t<OpenMP>::device_t(const device_t<OpenMP> &k);

  template <>
  device_t<OpenMP>::device_t(const int platform, const int device);

  template <>
  device_t<OpenMP>& device_t<OpenMP>::operator = (const device_t<OpenMP> &k);

  template <>
  void device_t<OpenMP>::setup(const int platform, const int device);

  template <>
  void device_t<OpenMP>::flush();

  template <>
  void device_t<OpenMP>::finish();

  template <>
  stream device_t<OpenMP>::genStream();

  template <>
  void device_t<OpenMP>::freeStream(stream s);

  template <>
  kernel_v* device_t<OpenMP>::buildKernelFromSource(const std::string &filename,
                                                    const std::string &functionName_,
                                                    const kernelInfo &info_);

  template <>
  kernel_v* device_t<OpenMP>::buildKernelFromBinary(const std::string &filename,
                                                    const std::string &functionName_);

  template <>
  memory_v* device_t<OpenMP>::malloc(const size_t bytes,
                                     void *source);

  template <>
  int device_t<OpenMP>::simdWidth();
  //==================================
};

#endif
