#if OCCA_CUDA_ENABLED
#  ifndef OCCA_CUDA_HEADER
#  define OCCA_CUDA_HEADER

#include "occa/base.hpp"
#include "occa/library.hpp"

#include "occa/defines.hpp"

#include <cuda.h>

namespace occa {
  //---[ Data Structs ]---------------
  struct CUDAKernelData_t {
    CUdevice   device;
    CUcontext  context;
    CUmodule   module;
    CUfunction function;

    void **vArgs;
  };

  struct CUDADeviceData_t {
    CUdevice  device;
    CUcontext context;
    bool p2pEnabled;
  };

  struct CUDATextureData_t {
    CUarray array;
    CUsurfObject surface;
  };
  //==================================


  //---[ Helper Functions ]-----------
  namespace cuda {
    extern bool isInitialized;

    void init();

    int getDeviceCount();

    CUdevice getDevice(const int id);

    uintptr_t getDeviceMemorySize(CUdevice device);

    std::string getDeviceListInfo();

    void enablePeerToPeer(CUcontext context);

    void checkPeerToPeer(CUdevice destDevice,
                         CUdevice srcDevice);

    void peerToPeerMemcpy(CUdevice destDevice,
                          CUcontext destContext,
                          CUdeviceptr destMemory,
                          CUdevice srcDevice,
                          CUcontext srcContext,
                          CUdeviceptr srcMemory,
                          const uintptr_t bytes,
                          CUstream usingStream);


    void asyncPeerToPeerMemcpy(CUdevice destDevice,
                               CUcontext destContext,
                               CUdeviceptr destMemory,
                               CUdevice srcDevice,
                               CUcontext srcContext,
                               CUdeviceptr srcMemory,
                               const uintptr_t bytes,
                               CUstream usingStream);

    void peerToPeerMemcpy(CUdevice destDevice,
                          CUcontext destContext,
                          CUdeviceptr destMemory,
                          CUdevice srcDevice,
                          CUcontext srcContext,
                          CUdeviceptr srcMemory,
                          const uintptr_t bytes,
                          CUstream usingStream,
                          const bool isAsync);
  }

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
  void* kernel_t<CUDA>::getKernelHandle();

  template <>
  void* kernel_t<CUDA>::getProgramHandle();

  template <>
  std::string kernel_t<CUDA>::fixBinaryName(const std::string &filename);

  template <>
  kernel_t<CUDA>* kernel_t<CUDA>::buildFromSource(const std::string &filename,
                                                  const std::string &functionName,
                                                  const kernelInfo &info_);

  template <>
  kernel_t<CUDA>* kernel_t<CUDA>::buildFromBinary(const std::string &filename,
                                                  const std::string &functionName);

  template <>
  kernel_t<CUDA>* kernel_t<CUDA>::loadFromLibrary(const char *cache,
                                                  const std::string &functionName);

  template <>
  uintptr_t kernel_t<CUDA>::maximumInnerDimSize();

  template <>
  int kernel_t<CUDA>::preferredDimSize();

  template <>
  void kernel_t<CUDA>::runFromArguments(const int kArgc, const kernelArg *kArgs);

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
  void memory_t<CUDA>::copyFrom(const void *src,
                                const uintptr_t bytes,
                                const uintptr_t offset);

  template <>
  void memory_t<CUDA>::copyFrom(const memory_v *src,
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
  void memory_t<CUDA>::asyncCopyFrom(const void *src,
                                     const uintptr_t bytes,
                                     const uintptr_t offset);

  template <>
  void memory_t<CUDA>::asyncCopyFrom(const memory_v *src,
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
  void memory_t<CUDA>::mappedFree();

  template <>
  void memory_t<CUDA>::mappedDetach();

  template <>
  void memory_t<CUDA>::free();

  template <>
  void memory_t<CUDA>::detach();
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<CUDA>::device_t();

  template <>
  device_t<CUDA>::device_t(const device_t<CUDA> &k);

  template <>
  device_t<CUDA>& device_t<CUDA>::operator = (const device_t<CUDA> &k);

  template <>
  void* device_t<CUDA>::getContextHandle();

  template <>
  void device_t<CUDA>::setup(argInfoMap &aim);

  template <>
  void device_t<CUDA>::addOccaHeadersToInfo(kernelInfo &info_);

  template <>
  std::string device_t<CUDA>::getInfoSalt(const kernelInfo &info_);

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
  void device_t<CUDA>::flush();

  template <>
  void device_t<CUDA>::finish();

  template <>
  void device_t<CUDA>::waitFor(streamTag tag);

  template <>
  stream_t device_t<CUDA>::createStream();

  template <>
  void device_t<CUDA>::freeStream(stream_t s);

  template <>
  stream_t device_t<CUDA>::wrapStream(void *handle_);

  template <>
  streamTag device_t<CUDA>::tagStream();

  template <>
  double device_t<CUDA>::timeBetween(const streamTag &startTag, const streamTag &endTag);

  template <>
  std::string device_t<CUDA>::fixBinaryName(const std::string &filename);

  template <>
  kernel_v* device_t<CUDA>::buildKernelFromSource(const std::string &filename,
                                                  const std::string &functionName,
                                                  const kernelInfo &info_);

  template <>
  kernel_v* device_t<CUDA>::buildKernelFromBinary(const std::string &filename,
                                                  const std::string &functionName);

  template <>
  void device_t<CUDA>::cacheKernelInLibrary(const std::string &filename,
                                            const std::string &functionName,
                                            const kernelInfo &info_);

  template <>
  kernel_v* device_t<CUDA>::loadKernelFromLibrary(const char *cache,
                                                  const std::string &functionName);

  template <>
  memory_v* device_t<CUDA>::wrapMemory(void *handle_,
                                       const uintptr_t bytes);

  template <>
  memory_v* device_t<CUDA>::wrapTexture(void *handle_,
                                        const int dim, const occa::dim &dims,
                                        occa::formatType type, const int permissions);

  template <>
  memory_v* device_t<CUDA>::malloc(const uintptr_t bytes,
                                   void *src);

  template <>
  memory_v* device_t<CUDA>::textureAlloc(const int dim, const occa::dim &dims,
                                         void *src,
                                         occa::formatType type, const int permissions);

  template <>
  memory_v* device_t<CUDA>::mappedAlloc(const uintptr_t bytes,
                                        void *src);

  template <>
  uintptr_t device_t<CUDA>::memorySize();

  template <>
  void device_t<CUDA>::free();

  template <>
  int device_t<CUDA>::simdWidth();
  //==================================

  //---[ Error Handling ]-------------
  std::string cudaError(const CUresult errorCode);
  //==================================
}

#  endif
#endif
