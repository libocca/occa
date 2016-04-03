#if OCCA_OPENMP_ENABLED
#  ifndef OCCA_OPENMP_HEADER
#  define OCCA_OPENMP_HEADER

#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <fcntl.h>

#include "occa/base.hpp"
#include "occa/library.hpp"

#if (OCCA_OS & (LINUX_OS | OSX_OS))
#  include <dlfcn.h>
#else
#  include <windows.h>
#endif

namespace occa {
  //---[ Data Structs ]---------------
  struct OpenMPKernelData_t {
    void *dlHandle;
    handleFunction_t handle;

    void *vArgs[2*OCCA_MAX_ARGS];
  };

  struct OpenMPDeviceData_t {
    int vendor;
    bool supportsOpenMP;
    std::string OpenMPFlag;
  };
  //==================================


  //---[ Helper Functions ]-----------
  namespace omp {
    extern std::string notSupported;

    std::string baseCompilerFlag(const int vendor_);
    std::string compilerFlag(const int vendor_,
                             const std::string &compiler);
  }
  //==================================


  //---[ Kernel ]---------------------
  template <>
  kernel_t<OpenMP>::kernel_t();

  template <>
  kernel_t<OpenMP>::kernel_t(const kernel_t &k);

  template <>
  kernel_t<OpenMP>& kernel_t<OpenMP>::operator = (const kernel_t<OpenMP> &k);

  template <>
  kernel_t<OpenMP>::kernel_t(const kernel_t<OpenMP> &k);

  template <>
  void* kernel_t<OpenMP>::getKernelHandle();

  template <>
  std::string kernel_t<OpenMP>::fixBinaryName(const std::string &filename);

  template <>
  kernel_t<OpenMP>* kernel_t<OpenMP>::buildFromSource(const std::string &filename,
                                                      const std::string &functionName,
                                                      const kernelInfo &info_);

  template <>
  kernel_t<OpenMP>* kernel_t<OpenMP>::buildFromBinary(const std::string &filename,
                                                      const std::string &functionName);

  template <>
  kernel_t<OpenMP>* kernel_t<OpenMP>::loadFromLibrary(const char *cache,
                                                      const std::string &functionName);

  template <>
  uintptr_t kernel_t<OpenMP>::maximumInnerDimSize();

  template <>
  int kernel_t<OpenMP>::preferredDimSize();

  template <>
  void kernel_t<OpenMP>::runFromArguments(const int kArgc, const kernelArg *kArgs);

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
  void* memory_t<OpenMP>::getMemoryHandle();

  template <>
  void* memory_t<OpenMP>::getTextureHandle();

  template <>
  void memory_t<OpenMP>::copyFrom(const void *src,
                                  const uintptr_t bytes,
                                  const uintptr_t offset);

  template <>
  void memory_t<OpenMP>::copyFrom(const memory_v *src,
                                  const uintptr_t bytes,
                                  const uintptr_t destOffset,
                                  const uintptr_t srcOffset);

  template <>
  void memory_t<OpenMP>::copyTo(void *dest,
                                const uintptr_t bytes,
                                const uintptr_t destOffset);

  template <>
  void memory_t<OpenMP>::copyTo(memory_v *dest,
                                const uintptr_t bytes,
                                const uintptr_t srcOffset,
                                const uintptr_t offset);

  template <>
  void memory_t<OpenMP>::asyncCopyFrom(const void *src,
                                       const uintptr_t bytes,
                                       const uintptr_t destOffset);

  template <>
  void memory_t<OpenMP>::asyncCopyFrom(const memory_v *src,
                                       const uintptr_t bytes,
                                       const uintptr_t srcOffset,
                                       const uintptr_t offset);

  template <>
  void memory_t<OpenMP>::asyncCopyTo(void *dest,
                                     const uintptr_t bytes,
                                     const uintptr_t offset);

  template <>
  void memory_t<OpenMP>::asyncCopyTo(memory_v *dest,
                                     const uintptr_t bytes,
                                     const uintptr_t destOffset,
                                     const uintptr_t srcOffset);

  template <>
  void memory_t<OpenMP>::mappedFree();

  template <>
  void memory_t<OpenMP>::free();
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<OpenMP>::device_t();

  template <>
  device_t<OpenMP>::device_t(const device_t<OpenMP> &k);

  template <>
  device_t<OpenMP>& device_t<OpenMP>::operator = (const device_t<OpenMP> &k);

  template <>
  void* device_t<OpenMP>::getContextHandle();

  template <>
  void device_t<OpenMP>::setup(argInfoMap &aim);

  template <>
  void device_t<OpenMP>::addOccaHeadersToInfo(kernelInfo &info_);

  template <>
  std::string device_t<OpenMP>::getInfoSalt(const kernelInfo &info_);

  template <>
  deviceIdentifier device_t<OpenMP>::getIdentifier() const;

  template <>
  void device_t<OpenMP>::getEnvironmentVariables();

  template <>
  void device_t<OpenMP>::appendAvailableDevices(std::vector<device> &dList);

  template <>
  void device_t<OpenMP>::setCompiler(const std::string &compiler_);

  template <>
  void device_t<OpenMP>::setCompilerEnvScript(const std::string &compilerEnvScript_);

  template <>
  void device_t<OpenMP>::setCompilerFlags(const std::string &compilerFlags_);

  template <>
  void device_t<OpenMP>::flush();

  template <>
  void device_t<OpenMP>::finish();

  template <>
  void device_t<OpenMP>::waitFor(streamTag tag);

  template <>
  stream_t device_t<OpenMP>::createStream();

  template <>
  void device_t<OpenMP>::freeStream(stream_t s);

  template <>
  stream_t device_t<OpenMP>::wrapStream(void *handle_);

  template <>
  streamTag device_t<OpenMP>::tagStream();

  template <>
  double device_t<OpenMP>::timeBetween(const streamTag &startTag, const streamTag &endTag);

  template <>
  std::string device_t<OpenMP>::fixBinaryName(const std::string &filename);

  template <>
  kernel_v* device_t<OpenMP>::buildKernelFromSource(const std::string &filename,
                                                    const std::string &functionName,
                                                    const kernelInfo &info_);

  template <>
  kernel_v* device_t<OpenMP>::buildKernelFromBinary(const std::string &filename,
                                                    const std::string &functionName);

  template <>
  void device_t<OpenMP>::cacheKernelInLibrary(const std::string &filename,
                                              const std::string &functionName,
                                              const kernelInfo &info_);

  template <>
  kernel_v* device_t<OpenMP>::loadKernelFromLibrary(const char *cache,
                                                    const std::string &functionName);

  template <>
  memory_v* device_t<OpenMP>::wrapMemory(void *handle_,
                                         const uintptr_t bytes);

  template <>
  memory_v* device_t<OpenMP>::wrapTexture(void *handle_,
                                          const int dim, const occa::dim &dims,
                                          occa::formatType type, const int permissions);

  template <>
  memory_v* device_t<OpenMP>::malloc(const uintptr_t bytes,
                                     void *src);

  template <>
  memory_v* device_t<OpenMP>::textureAlloc(const int dim, const occa::dim &dims,
                                           void *src,
                                           occa::formatType type, const int permissions);

  template <>
  memory_v* device_t<OpenMP>::mappedAlloc(const uintptr_t bytes,
                                          void *src);

  template <>
  uintptr_t device_t<OpenMP>::memorySize();

  template <>
  void device_t<OpenMP>::free();

  template <>
  int device_t<OpenMP>::simdWidth();
  //==================================
}

#  endif
#endif
