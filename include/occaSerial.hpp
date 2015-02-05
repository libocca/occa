#ifndef OCCA_OPENMP_HEADER
#define OCCA_OPENMP_HEADER

#include <sys/sysctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <fcntl.h>

#include "occaBase.hpp"
#include "occaLibrary.hpp"

#include "occaKernelDefines.hpp"

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
#  include <dlfcn.h>
#else
#  include <windows.h>
#endif

namespace occa {
  //---[ Data Structs ]---------------
  struct SerialKernelData_t {
    void *dlHandle, *handle;
  };
  //==================================


  //---[ Helper Functions ]-----------
  namespace cpu {
    std::string getLSCPUField(std::string field);
    std::string getCPUINFOField(std::string field);

    std::string getProcessorName();
    int getCoreCount();
    int getProcessorFrequency();
    std::string getProcessorCacheSize(int level);

    std::string getDeviceListInfo();
  };
  //==================================


  //---[ Kernel ]---------------------
  template <>
  kernel_t<Serial>::kernel_t();

  template <>
  kernel_t<Serial>::kernel_t(const kernel_t &k);

  template <>
  kernel_t<Serial>& kernel_t<Serial>::operator = (const kernel_t<Serial> &k);

  template <>
  kernel_t<Serial>::kernel_t(const kernel_t<Serial> &k);

  template <>
  std::string kernel_t<Serial>::getCachedBinaryName(const std::string &filename,
                                                    kernelInfo &info_);

  template <>
  kernel_t<Serial>* kernel_t<Serial>::buildFromSource(const std::string &filename,
                                                      const std::string &functionName_,
                                                      const kernelInfo &info_);

  template <>
  kernel_t<Serial>* kernel_t<Serial>::buildFromBinary(const std::string &filename,
                                                      const std::string &functionName_);

  template <>
  kernel_t<Serial>* kernel_t<Serial>::loadFromLibrary(const char *cache,
                                                      const std::string &functionName_);

  template <>
  int kernel_t<Serial>::preferredDimSize();

  template <>
  double kernel_t<Serial>::timeTaken();

  template <>
  double kernel_t<Serial>::timeTakenBetween(void *start, void *end);

  template <>
  void kernel_t<Serial>::free();
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<Serial>::memory_t();

  template <>
  memory_t<Serial>::memory_t(const memory_t &m);

  template <>
  memory_t<Serial>& memory_t<Serial>::operator = (const memory_t &m);

  template <>
  void* memory_t<Serial>::getMemoryHandle();

  template <>
  void* memory_t<Serial>::getTextureHandle();

  template <>
  void memory_t<Serial>::copyFrom(const void *src,
                                  const uintptr_t bytes,
                                  const uintptr_t offset);

  template <>
  void memory_t<Serial>::copyFrom(const memory_v *src,
                                  const uintptr_t bytes,
                                  const uintptr_t destOffset,
                                  const uintptr_t srcOffset);

  template <>
  void memory_t<Serial>::copyTo(void *dest,
                                const uintptr_t bytes,
                                const uintptr_t destOffset);

  template <>
  void memory_t<Serial>::copyTo(memory_v *dest,
                                const uintptr_t bytes,
                                const uintptr_t srcOffset,
                                const uintptr_t offset);

  template <>
  void memory_t<Serial>::asyncCopyFrom(const void *src,
                                       const uintptr_t bytes,
                                       const uintptr_t destOffset);

  template <>
  void memory_t<Serial>::asyncCopyFrom(const memory_v *src,
                                       const uintptr_t bytes,
                                       const uintptr_t srcOffset,
                                       const uintptr_t offset);

  template <>
  void memory_t<Serial>::asyncCopyTo(void *dest,
                                     const uintptr_t bytes,
                                     const uintptr_t offset);

  template <>
  void memory_t<Serial>::asyncCopyTo(memory_v *dest,
                                     const uintptr_t bytes,
                                     const uintptr_t destOffset,
                                     const uintptr_t srcOffset);

  template <>
  void memory_t<Serial>::mappedFree();

  template <>
  void memory_t<Serial>::free();
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<Serial>::device_t();

  template <>
  device_t<Serial>::device_t(const device_t<Serial> &k);

  template <>
  device_t<Serial>& device_t<Serial>::operator = (const device_t<Serial> &k);

  template <>
  void device_t<Serial>::setup(argInfoMap &aim);

  template <>
  void device_t<Serial>::addOccaHeadersToInfo(kernelInfo &info_);

  template <>
  std::string device_t<Serial>::getInfoSalt(const kernelInfo &info_);

  template <>
  deviceIdentifier device_t<Serial>::getIdentifier() const;

  template <>
  void device_t<Serial>::getEnvironmentVariables();

  template <>
  void device_t<Serial>::appendAvailableDevices(std::vector<device> &dList);

  template <>
  void device_t<Serial>::setCompiler(const std::string &compiler_);

  template <>
  void device_t<Serial>::setCompilerEnvScript(const std::string &compilerEnvScript_);

  template <>
  void device_t<Serial>::setCompilerFlags(const std::string &compilerFlags_);

  template <>
  std::string& device_t<Serial>::getCompiler();

  template <>
  std::string& device_t<Serial>::getCompilerEnvScript();

  template <>
  std::string& device_t<Serial>::getCompilerFlags();

  template <>
  void device_t<Serial>::flush();

  template <>
  void device_t<Serial>::finish();

  template <>
  void device_t<Serial>::waitFor(tag tag_);

  template <>
  stream device_t<Serial>::createStream();

  template <>
  void device_t<Serial>::freeStream(stream s);

  template <>
  stream device_t<Serial>::wrapStream(void *handle_);

  template <>
  tag device_t<Serial>::tagStream();

  template <>
  double device_t<Serial>::timeBetween(const tag &startTag, const tag &endTag);

  template <>
  kernel_v* device_t<Serial>::buildKernelFromSource(const std::string &filename,
                                                    const std::string &functionName_,
                                                    const kernelInfo &info_);

  template <>
  kernel_v* device_t<Serial>::buildKernelFromBinary(const std::string &filename,
                                                    const std::string &functionName_);

  template <>
  void device_t<Serial>::cacheKernelInLibrary(const std::string &filename,
                                              const std::string &functionName_,
                                              const kernelInfo &info_);

  template <>
  kernel_v* device_t<Serial>::loadKernelFromLibrary(const char *cache,
                                                    const std::string &functionName_);

  template <>
  memory_v* device_t<Serial>::wrapMemory(void *handle_,
                                         const uintptr_t bytes);

  template <>
  memory_v* device_t<Serial>::wrapTexture(void *handle_,
                                          const int dim, const occa::dim &dims,
                                          occa::formatType type, const int permissions);

  template <>
  memory_v* device_t<Serial>::malloc(const uintptr_t bytes,
                                     void *src);

  template <>
  memory_v* device_t<Serial>::textureAlloc(const int dim, const occa::dim &dims,
                                           void *src,
                                           occa::formatType type, const int permissions);

  template <>
  memory_v* device_t<Serial>::mappedAlloc(const uintptr_t bytes,
                                          void *src);

  template <>
  void device_t<Serial>::free();

  template <>
  int device_t<Serial>::simdWidth();
  //==================================

#include "operators/occaFunctionPointerTypeDefs.hpp"
#include "operators/occaSerialKernelOperators.hpp"

};

#endif
