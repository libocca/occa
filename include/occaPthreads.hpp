#ifndef OCCA_OPENMP_HEADER
#define OCCA_OPENMP_HEADER

#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <dlfcn.h>
#include <fcntl.h>

#include "occaBase.hpp"

#include "occaKernelDefines.hpp"

#include <pthread.h>

namespace occa {
  //---[ Data Structs ]---------------
  struct PthreadsKernelData_t {
    void *dlHandle, *handle;
  };
  //==================================


  //---[ Kernel ]---------------------
  OCCA_OPENMP_FUNCTION_POINTER_TYPEDEFS;

  template <>
  kernel_t<Pthreads>::kernel_t();

  template <>
  kernel_t<Pthreads>::kernel_t(const kernel_t &k);

  template <>
  kernel_t<Pthreads>& kernel_t<Pthreads>::operator = (const kernel_t<Pthreads> &k);

  template <>
  kernel_t<Pthreads>::kernel_t(const kernel_t<Pthreads> &k);

  template <>
  kernel_t<Pthreads>* kernel_t<Pthreads>::buildFromSource(const std::string &filename,
                                                      const std::string &functionName_,
                                                      const kernelInfo &info_);

  template <>
  kernel_t<Pthreads>* kernel_t<Pthreads>::buildFromBinary(const std::string &filename,
                                                      const std::string &functionName_);

  template <>
  int kernel_t<Pthreads>::preferredDimSize();

  template <>
  double kernel_t<Pthreads>::timeTaken();

  template <>
  void kernel_t<Pthreads>::free();
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<Pthreads>::memory_t();

  template <>
  memory_t<Pthreads>::memory_t(const memory_t &m);

  template <>
  memory_t<Pthreads>& memory_t<Pthreads>::operator = (const memory_t &m);

  template <>
  void memory_t<Pthreads>::copyFrom(const void *source,
                                  const size_t bytes,
                                  const size_t offset);

  template <>
  void memory_t<Pthreads>::copyFrom(const memory_v *source,
                                  const size_t bytes,
                                  const size_t offset);

  template <>
  void memory_t<Pthreads>::copyTo(void *dest,
                                const size_t bytes,
                                const size_t offset);

  template <>
  void memory_t<Pthreads>::copyTo(memory_v *dest,
                                const size_t bytes,
                                const size_t offset);

  template <>
  void memory_t<Pthreads>::asyncCopyFrom(const void *source,
                                       const size_t bytes,
                                       const size_t offset);

  template <>
  void memory_t<Pthreads>::asyncCopyFrom(const memory_v *source,
                                       const size_t bytes,
                                       const size_t offset);

  template <>
  void memory_t<Pthreads>::asyncCopyTo(void *dest,
                                     const size_t bytes,
                                     const size_t offset);

  template <>
  void memory_t<Pthreads>::asyncCopyTo(memory_v *dest,
                                     const size_t bytes,
                                     const size_t offset);

  template <>
  void memory_t<Pthreads>::free();
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<Pthreads>::device_t();

  template <>
  device_t<Pthreads>::device_t(const device_t<Pthreads> &k);

  template <>
  device_t<Pthreads>::device_t(const int platform, const int device);

  template <>
  device_t<Pthreads>& device_t<Pthreads>::operator = (const device_t<Pthreads> &k);

  template <>
  void device_t<Pthreads>::setup(const int platform, const int device);

  template <>
  void device_t<Pthreads>::getEnvironmentVariables();

  template <>
  void device_t<Pthreads>::setCompiler(const std::string &compiler);

  template <>
  void device_t<Pthreads>::setCompilerFlags(const std::string &compilerFlags);

  template <>
  void device_t<Pthreads>::flush();

  template <>
  void device_t<Pthreads>::finish();

  template <>
  stream device_t<Pthreads>::genStream();

  template <>
  void device_t<Pthreads>::freeStream(stream s);

  template <>
  kernel_v* device_t<Pthreads>::buildKernelFromSource(const std::string &filename,
                                                    const std::string &functionName_,
                                                    const kernelInfo &info_);

  template <>
  kernel_v* device_t<Pthreads>::buildKernelFromBinary(const std::string &filename,
                                                    const std::string &functionName_);

  template <>
  memory_v* device_t<Pthreads>::malloc(const size_t bytes,
                                     void *source);

  template <>
  int device_t<Pthreads>::simdWidth();
  //==================================
};

#endif
