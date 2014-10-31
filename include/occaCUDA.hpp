#if OCCA_CUDA_ENABLED
#  ifndef OCCA_CUDA_HEADER
#  define OCCA_CUDA_HEADER

#include "occaBase.hpp"
#include "occaLibrary.hpp"

#include "occaKernelDefines.hpp"

#include "occaDefines.hpp"

#include <cuda.h>

namespace occa {
  //---[ Data Structs ]---------------
  struct CUDAKernelData_t {
    CUdevice   device;
    CUcontext  context;
    CUmodule   module;
    CUfunction function;
  };

  struct CUDADeviceData_t {
    CUdevice  device;
    CUcontext context;
  };

  struct CUDATextureData_t {
    CUarray array;
    CUsurfObject surface;
  };
  //==================================


  //---[ Helper Functions ]-----------
  namespace cuda {
    static bool isNotInitialized = true;

    void init();
  };

  extern const CUarray_format cudaFormats[8];

  template <>
  void* formatType::format<occa::CUDA>() const;

  extern const int CUDA_ADDRESS_NONE;
  extern const int CUDA_ADDRESS_CLAMP;
  //==================================


  //---[ Kernel ]---------------------
  template <>
  kernel_t<CUDA>::kernel_t();

  template <>
  kernel_t<CUDA>::kernel_t(const kernel_t &k);

  template <>
  kernel_t<CUDA>& kernel_t<CUDA>::operator = (const kernel_t<CUDA> &k);

  template <>
  kernel_t<CUDA>::kernel_t(const kernel_t<CUDA> &k);

  template <>
  std::string kernel_t<CUDA>::getCachedBinaryName(const std::string &filename,
                                                  kernelInfo &info_);

  template <>
  kernel_t<CUDA>* kernel_t<CUDA>::buildFromSource(const std::string &filename,
                                                  const std::string &functionName_,
                                                  const kernelInfo &info_);

  template <>
  kernel_t<CUDA>* kernel_t<CUDA>::buildFromBinary(const std::string &filename,
                                                  const std::string &functionName_);

  template <>
  kernel_t<CUDA>* kernel_t<CUDA>::loadFromLibrary(const char *cache,
                                                  const std::string &functionName_);

  template <>
  int kernel_t<CUDA>::preferredDimSize();

  template <>
  double kernel_t<CUDA>::timeTaken();

  template <>
  double kernel_t<CUDA>::timeTakenBetween(void *start, void *end);

  template <>
  void kernel_t<CUDA>::free();
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<CUDA>::memory_t();

  template <>
  memory_t<CUDA>::memory_t(const memory_t &m);

  template <>
  memory_t<CUDA>& memory_t<CUDA>::operator = (const memory_t &m);

  template <>
  void* memory_t<CUDA>::getMemoryHandle();

  template <>
  void* memory_t<CUDA>::getTextureHandle();

  template <>
  void memory_t<CUDA>::copyFrom(const void *source,
                                const uintptr_t bytes,
                                const uintptr_t offset);

  template <>
  void memory_t<CUDA>::copyFrom(const memory_v *source,
                                const uintptr_t bytes,
                                const uintptr_t destOffset,
                                const uintptr_t srcOffset);

  template <>
  void memory_t<CUDA>::copyTo(void *dest,
                              const uintptr_t bytes,
                              const uintptr_t offset);

  template <>
  void memory_t<CUDA>::copyTo(memory_v *dest,
                              const uintptr_t bytes,
                              const uintptr_t destOffset,
                              const uintptr_t srcOffset);

  template <>
  void memory_t<CUDA>::asyncCopyFrom(const void *source,
                                     const uintptr_t bytes,
                                     const uintptr_t offset);

  template <>
  void memory_t<CUDA>::asyncCopyFrom(const memory_v *source,
                                     const uintptr_t bytes,
                                     const uintptr_t destOffset,
                                     const uintptr_t srcOffset);

  template <>
  void memory_t<CUDA>::asyncCopyTo(void *dest,
                                   const uintptr_t bytes,
                                   const uintptr_t offset);

  template <>
  void memory_t<CUDA>::asyncCopyTo(memory_v *dest,
                                   const uintptr_t bytes,
                                   const uintptr_t destOffset,
                                   const uintptr_t srcOffset);

  template <>
  void memory_t<CUDA>::free();
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<CUDA>::device_t();

  template <>
  device_t<CUDA>::device_t(const device_t<CUDA> &k);

  template <>
  device_t<CUDA>::device_t(const int arg1, const int arg2);

  template <>
  device_t<CUDA>& device_t<CUDA>::operator = (const device_t<CUDA> &k);

  template <>
  void device_t<CUDA>::setup(const int device, const int unusedArg);

  template <>
  deviceIdentifier device_t<CUDA>::getIdentifier() const;

  template <>
  void device_t<CUDA>::getEnvironmentVariables();

  template <>
  void device_t<CUDA>::appendAvailableDevices(std::vector<device> &dList);

  template <>
  void device_t<CUDA>::setCompiler(const std::string &compiler_);

  template <>
  void device_t<CUDA>::setCompilerEnvScript(const std::string &compilerEnvScript_);

  template <>
  void device_t<CUDA>::setCompilerFlags(const std::string &compilerFlags_);

  template <>
  std::string& device_t<CUDA>::getCompiler();

  template <>
  std::string& device_t<CUDA>::getCompilerEnvScript();

  template <>
  std::string& device_t<CUDA>::getCompilerFlags();

  template <>
  void device_t<CUDA>::flush();

  template <>
  void device_t<CUDA>::finish();

  template <>
  stream device_t<CUDA>::genStream();

  template <>
  void device_t<CUDA>::freeStream(stream s);

  template <>
  stream device_t<CUDA>::wrapStream(void *handle_);

  template <>
  tag device_t<CUDA>::tagStream();

  template <>
  double device_t<CUDA>::timeBetween(const tag &startTag, const tag &endTag);

  template <>
  kernel_v* device_t<CUDA>::buildKernelFromSource(const std::string &filename,
                                                  const std::string &functionName_,
                                                  const kernelInfo &info_);

  template <>
  kernel_v* device_t<CUDA>::buildKernelFromBinary(const std::string &filename,
                                                  const std::string &functionName_);

  template <>
  void device_t<CUDA>::cacheKernelInLibrary(const std::string &filename,
                                            const std::string &functionName_,
                                            const kernelInfo &info_);

  template <>
  kernel_v* device_t<CUDA>::loadKernelFromLibrary(const char *cache,
                                                  const std::string &functionName_);

  template <>
  memory_v* device_t<CUDA>::wrapMemory(void *handle_,
                                       const uintptr_t bytes);

  template <>
  memory_v* device_t<CUDA>::wrapTexture(void *handle_,
                                        const int dim, const occa::dim &dims,
                                        occa::formatType type, const int permissions);

  template <>
  memory_v* device_t<CUDA>::malloc(const uintptr_t bytes,
                                   void *source);

  template <>
  memory_v* device_t<CUDA>::talloc(const int dim, const occa::dim &dims,
                                   void *source,
                                   occa::formatType type, const int permissions);

  template <>
  void device_t<CUDA>::free();

  template <>
  int device_t<CUDA>::simdWidth();
  //==================================

#include "operators/occaCUDAKernelOperators.hpp"

  //---[ Error Handling ]-------------
  std::string cudaError(const CUresult errorCode);
  //==================================
};

#  endif
#endif
