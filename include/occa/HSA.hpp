#if OCCA_HSA_ENABLED
#  ifndef OCCA_HSA_HEADER
#  define OCCA_HSA_HEADER

#include "occa/base.hpp"
#include "occa/library.hpp"

#include "occa/defines.hpp"

#if   (OCCA_OS & LINUX_OS)
#elif (OCCA_OS & OSX_OS)
#else
#endif

namespace occa {
  //---[ Data Structs ]---------------
  struct HSAKernelData_t {
    // int platform, device;

    // cl_platform_id platformID;
    // cl_device_id   deviceID;
    // cl_context     context;
    // cl_program     program;
    // cl_kernel      kernel;
  };

  struct HSADeviceData_t {
    // int platform, device;

    // cl_platform_id platformID;
    // cl_device_id   deviceID;
    // cl_context     context;
  };
  //==================================


  //---[ Helper Functions ]-----------
  namespace hsa {
    extern bool initialized;
    extern hsa_status_t

    void init();

    struct getDeviceInfo_t {
      int id, chosenId;
      hsa_agent_t *agent;
    };

    hsa_status_t getDeviceCount_h(hsa_agent_t agent, void *count_){
    hsa_status_t getDevice_h(hsa_agent_t agent, void *data_);
  };

  extern const cl_channel_type clFormats[8];

  template <>
  void* formatType::format<occa::HSA>() const;
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
                                    const uintptr_t destOffset);

  template <>
  void memory_t<HSA>::asyncCopyFrom(const memory_v *src,
                                    const uintptr_t bytes,
                                    const uintptr_t srcOffset,
                                    const uintptr_t offset);

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
  void device_t<HSA>::free();

  template <>
  int device_t<HSA>::simdWidth();
  //==================================

#include "operators/occaHSAKernelOperators.hpp"

  //---[ Error Handling ]-------------
  std::string hsaError(int e);
  //==================================
}

#  endif
#endif
