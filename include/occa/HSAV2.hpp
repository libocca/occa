#if OCCA_HSA_ENABLED
#  ifndef OCCA_HSA_HEADER
#  define OCCA_HSA_HEADER

#include "occa/base.hpp"
#include "occa/library.hpp"

#include "occa/defines.hpp"

#include <hsa.h>

// HSA agent docs: http://www.hsafoundation.com/html/HSA_Library.htm#Runtime/Topics/01_Intro/initialization_and_agent_discovery.htm
// HSA queue docs: http://www.hsafoundation.com/html/HSA_Library.htm#Runtime/Topics/01_Intro/queues_and_AQL_packets.htm%3FTocPath%3DHSA%2520Runtime%2520Programmer%25E2%2580%2599s%2520Reference%2520Manual%2520Version%25201.0%2520%7CChapter%25201.%2520Introduction%7CProgramming%2520Model%7C_____2
// 

namespace occa {
  //---[ Data Structs ]---------------
  struct HSAKernelData_t {

    hsa_agent_t kernel_agent;
    hsa_queue_t *queue;
    hsa_region_t kernarg_region;

    CUcontext  context;
    CUmodule   module;
    CUfunction function;

    void *vArgs[2*OCCA_MAX_ARGS];
  };

  struct HSADeviceData_t {

    hsa_agent_t agent;

    CUdevice  device;
    CUcontext context;
    bool p2pEnabled;
  };

  struct HSATextureData_t {
    CUarray array;
    CUsurfObject surface;
  };
  //==================================


  //---[ Helper Functions ]-----------
  namespace hsa {
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

  extern const CUarray_format hsaFormats[8];

  template <>
  void* formatType::format<occa::HSA>() const;

  extern const int HSA_ADDRESS_NONE;
  extern const int HSA_ADDRESS_CLAMP;
  //==================================


  //---[ Kernel ]---------------------
  template <>
  kernel_t<HSA>::kernel_t();

  template <>
  kernel_t<HSA>::kernel_t(const kernel_t &k);

  template <>
  kernel_t<HSA>& kernel_t<HSA>::operator = (const kernel_t<HSA> &k);

  template <>
  kernel_t<HSA>::kernel_t(const kernel_t<HSA> &k);

  template <>
  void* kernel_t<HSA>::getKernelHandle();

  template <>
  void* kernel_t<HSA>::getProgramHandle();

  template <>
  std::string kernel_t<HSA>::fixBinaryName(const std::string &filename);

  template <>
  kernel_t<HSA>* kernel_t<HSA>::buildFromSource(const std::string &filename,
                                                  const std::string &functionName,
                                                  const kernelInfo &info_);

  template <>
  kernel_t<HSA>* kernel_t<HSA>::buildFromBinary(const std::string &filename,
                                                  const std::string &functionName);

  template <>
  kernel_t<HSA>* kernel_t<HSA>::loadFromLibrary(const char *cache,
                                                  const std::string &functionName);

  template <>
  uintptr_t kernel_t<HSA>::maximumInnerDimSize();

  template <>
  int kernel_t<HSA>::preferredDimSize();

  template <>
  void kernel_t<HSA>::free();
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<HSA>::memory_t();

  template <>
  memory_t<HSA>::memory_t(const memory_t &m);

  template <>
  memory_t<HSA>& memory_t<HSA>::operator = (const memory_t &m);

  template <>
  void* memory_t<HSA>::getMemoryHandle();

  template <>
  void* memory_t<HSA>::getTextureHandle();

  template <>
  void memory_t<HSA>::copyFrom(const void *src,
                                const uintptr_t bytes,
                                const uintptr_t offset);

  template <>
  void memory_t<HSA>::copyFrom(const memory_v *src,
                                const uintptr_t bytes,
                                const uintptr_t destOffset,
                                const uintptr_t srcOffset);

  template <>
  void memory_t<HSA>::copyTo(void *dest,
                              const uintptr_t bytes,
                              const uintptr_t offset);

  template <>
  void memory_t<HSA>::copyTo(memory_v *dest,
                              const uintptr_t bytes,
                              const uintptr_t destOffset,
                              const uintptr_t srcOffset);

  template <>
  void memory_t<HSA>::asyncCopyFrom(const void *src,
                                     const uintptr_t bytes,
                                     const uintptr_t offset);

  template <>
  void memory_t<HSA>::asyncCopyFrom(const memory_v *src,
                                     const uintptr_t bytes,
                                     const uintptr_t destOffset,
                                     const uintptr_t srcOffset);

  template <>
  void memory_t<HSA>::asyncCopyTo(void *dest,
                                   const uintptr_t bytes,
                                   const uintptr_t offset);

  template <>
  void memory_t<HSA>::asyncCopyTo(memory_v *dest,
                                   const uintptr_t bytes,
                                   const uintptr_t destOffset,
                                   const uintptr_t srcOffset);

  template <>
  void memory_t<HSA>::mappedFree();

  template <>
  void memory_t<HSA>::free();
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<HSA>::device_t();

  template <>
  device_t<HSA>::device_t(const device_t<HSA> &k);

  template <>
  device_t<HSA>& device_t<HSA>::operator = (const device_t<HSA> &k);

  template <>
  void* device_t<HSA>::getContextHandle();

  template <>
  void device_t<HSA>::setup(argInfoMap &aim);

  template <>
  void device_t<HSA>::addOccaHeadersToInfo(kernelInfo &info_);

  template <>
  std::string device_t<HSA>::getInfoSalt(const kernelInfo &info_);

  template <>
  deviceIdentifier device_t<HSA>::getIdentifier() const;

  template <>
  void device_t<HSA>::getEnvironmentVariables();

  template <>
  void device_t<HSA>::appendAvailableDevices(std::vector<device> &dList);

  template <>
  void device_t<HSA>::setCompiler(const std::string &compiler_);

  template <>
  void device_t<HSA>::setCompilerEnvScript(const std::string &compilerEnvScript_);

  template <>
  void device_t<HSA>::setCompilerFlags(const std::string &compilerFlags_);

  template <>
  void device_t<HSA>::flush();

  template <>
  void device_t<HSA>::finish();

  template <>
  void device_t<HSA>::waitFor(streamTag tag);

  template <>
  stream_t device_t<HSA>::createStream();

  template <>
  void device_t<HSA>::freeStream(stream_t s);

  template <>
  stream_t device_t<HSA>::wrapStream(void *handle_);

  template <>
  streamTag device_t<HSA>::tagStream();

  template <>
  double device_t<HSA>::timeBetween(const streamTag &startTag, const streamTag &endTag);

  template <>
  std::string device_t<HSA>::fixBinaryName(const std::string &filename);

  template <>
  kernel_v* device_t<HSA>::buildKernelFromSource(const std::string &filename,
                                                  const std::string &functionName,
                                                  const kernelInfo &info_);

  template <>
  kernel_v* device_t<HSA>::buildKernelFromBinary(const std::string &filename,
                                                  const std::string &functionName);

  template <>
  void device_t<HSA>::cacheKernelInLibrary(const std::string &filename,
                                            const std::string &functionName,
                                            const kernelInfo &info_);

  template <>
  kernel_v* device_t<HSA>::loadKernelFromLibrary(const char *cache,
                                                  const std::string &functionName);

  template <>
  memory_v* device_t<HSA>::wrapMemory(void *handle_,
                                       const uintptr_t bytes);

  template <>
  memory_v* device_t<HSA>::wrapTexture(void *handle_,
                                        const int dim, const occa::dim &dims,
                                        occa::formatType type, const int permissions);

  template <>
  memory_v* device_t<HSA>::malloc(const uintptr_t bytes,
                                   void *src);

  template <>
  memory_v* device_t<HSA>::textureAlloc(const int dim, const occa::dim &dims,
                                         void *src,
                                         occa::formatType type, const int permissions);

  template <>
  memory_v* device_t<HSA>::mappedAlloc(const uintptr_t bytes,
                                        void *src);

  template <>
  uintptr_t device_t<HSA>::memorySize();

  template <>
  void device_t<HSA>::free();

  template <>
  int device_t<HSA>::simdWidth();
  //==================================

#include "occa/operators/HSAKernelOperators.hpp"

  //---[ Error Handling ]-------------
  std::string hsaError(const HSAresult errorCode);
  //==================================
}

#  endif
#endif
