#if OCCA_HSA_ENABLED

#include "occa/HSA.hpp"

namespace occa {
  //---[ Helper Functions ]-----------
  namespace hsa {
    bool initialized = false;

    void init(){
      if(initialized)
        return;

      // 1) make sure that HSA is initialized
      OCCA_HSA_CHECK("Initializing HSA",
                     hsa_init());

      initialized = true;
    }

    hsa_status_t getDeviceCount_h(hsa_agent_t agent, void *count_){
      OCCA_CHECK(count_ != NULL,
                 "HSA : getDeviceCount argument [count] is NULL");

      int &count = *((int*) count_);
      ++count;

      return HSA_STATUS_SUCCESS;
    }

    hsa_status_t getDevice_h(hsa_agent_t agent, void *data_){
      OCCA_CHECK(data_ != NULL,
                 "HSA : getDevice argument [data] is NULL");

      getDeviceInfo_t &data = *((getDeviceInfo_t*) data_);

      if(chosenID

      // hsa_device_type_t deviceType;

      // OCCA_HSA_CHECK(hsa_agent_get_info(agent,
      //                                   HSA_AGENT_INFO_DEVICE,
      //                                   &deviceType),
      //                "Getting device type");

      hsa_agent_t *agent2 = (hsa_agent_t*) data;

      if(type == (HSA_DEVICE_TYPE_CPU |
                  HSA_DEVICE_TYPE_GPU |
                  HSA_DEVICE_TYPE_DSP)
    }
  };
  //==================================


  //---[ Kernel ]---------------------
  template <>
  kernel_t<HSA>::kernel_t(){}

  template <>
  kernel_t<HSA>::kernel_t(const kernel_t<HSA> &k){}

  template <>
  kernel_t<HSA>& kernel_t<HSA>::operator = (const kernel_t<HSA> &k){}

  template <>
  kernel_t<HSA>::~kernel_t(){}

  template <>
  std::string kernel_t<HSA>::fixBinaryName(const std::string &filename){
    return filename;
  }

  template <>
  std::string kernel_t<HSA>::getCachedBinaryName(const std::string &filename,
                                                 kernelInfo &info_){}

  template <>
  kernel_t<HSA>* kernel_t<HSA>::buildFromSource(const std::string &filename,
                                                const std::string &functionName,
                                                const kernelInfo &info_){}

  template <>
  kernel_t<HSA>* kernel_t<HSA>::buildFromBinary(const std::string &filename,
                                                const std::string &functionName){}

  template <>
  kernel_t<HSA>* kernel_t<HSA>::loadFromLibrary(const char *cache,
                                                const std::string &functionName){}

  template <>
  uintptr_t kernel_t<HSA>::maximumInnerDimSize(){
  }

  template <>
  int kernel_t<HSA>::preferredDimSize(){}

#include "operators/occaHSAKernelOperators.cpp"

  template <>
  void kernel_t<HSA>::free(){}
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<HSA>::memory_t(){}

  template <>
  memory_t<HSA>::memory_t(const memory_t<HSA> &m){}

  template <>
  memory_t<HSA>& memory_t<HSA>::operator = (const memory_t<HSA> &m){}

  template <>
  memory_t<HSA>::~memory_t(){}

  template <>
  void* memory_t<HSA>::getMemoryHandle(){}

  template <>
  void* memory_t<HSA>::getTextureHandle(){}

  template <>
  void memory_t<HSA>::copyFrom(const void *src,
                               const uintptr_t bytes,
                               const uintptr_t offset){}

  template <>
  void memory_t<HSA>::copyFrom(const memory_v *src,
                               const uintptr_t bytes,
                               const uintptr_t destOffset,
                               const uintptr_t srcOffset){}

  template <>
  void memory_t<HSA>::copyTo(void *dest,
                             const uintptr_t bytes,
                             const uintptr_t offset){}

  template <>
  void memory_t<HSA>::copyTo(memory_v *dest,
                             const uintptr_t bytes,
                             const uintptr_t destOffset,
                             const uintptr_t srcOffset){}

  template <>
  void memory_t<HSA>::asyncCopyFrom(const void *src,
                                    const uintptr_t bytes,
                                    const uintptr_t offset){}

  template <>
  void memory_t<HSA>::asyncCopyFrom(const memory_v *src,
                                    const uintptr_t bytes,
                                    const uintptr_t destOffset,
                                    const uintptr_t srcOffset){}

  template <>
  void memory_t<HSA>::asyncCopyTo(void *dest,
                                  const uintptr_t bytes,
                                  const uintptr_t offset){}

  template <>
  void memory_t<HSA>::asyncCopyTo(memory_v *dest,
                                  const uintptr_t bytes,
                                  const uintptr_t destOffset,
                                  const uintptr_t srcOffset){}

  template <>
  void memory_t<HSA>::mappedFree(){}

  template <>
  void memory_t<HSA>::free(){}
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<HSA>::device_t() {}

  template <>
  device_t<HSA>::device_t(const device_t<HSA> &d){}

  template <>
  device_t<HSA>& device_t<HSA>::operator = (const device_t<HSA> &d){}

  template <>
  void device_t<HSA>::setup(argInfoMap &aim){}

  template <>
  void device_t<HSA>::addOccaHeadersToInfo(kernelInfo &info_){
    info_.mode = HSA;
  }

  template <>
  std::string device_t<HSA>::getInfoSalt(const kernelInfo &info_){}

  template <>
  deviceIdentifier device_t<HSA>::getIdentifier() const {}

  template <>
  void device_t<HSA>::getEnvironmentVariables(){}

  template <>
  void device_t<HSA>::appendAvailableDevices(std::vector<device> &dList){}

  template <>
  void device_t<HSA>::setCompiler(const std::string &compiler_){}

  template <>
  void device_t<HSA>::setCompilerEnvScript(const std::string &compilerEnvScript_){}

  template <>
  void device_t<HSA>::setCompilerFlags(const std::string &compilerFlags_){}

  template <>
  void device_t<HSA>::flush(){}

  template <>
  void device_t<HSA>::finish(){}

  template <>
  bool device_t<HSA>::fakesUva(){}

  template <>
  void device_t<HSA>::waitFor(streamTag tag){}

  template <>
  stream_t device_t<HSA>::createStream(){}

  template <>
  void device_t<HSA>::freeStream(stream_t s){}

  template <>
  stream_t device_t<HSA>::wrapStream(void *handle_){}

  template <>
  streamTag device_t<HSA>::tagStream(){}

  template <>
  double device_t<HSA>::timeBetween(const streamTag &startTag, const streamTag &endTag){}

  template <>
  kernel_v* device_t<HSA>::buildKernelFromSource(const std::string &filename,
                                                 const std::string &functionName,
                                                 const kernelInfo &info_){}

  template <>
  kernel_v* device_t<HSA>::buildKernelFromBinary(const std::string &filename,
                                                 const std::string &functionName){}

  template <>
  void device_t<HSA>::cacheKernelInLibrary(const std::string &filename,
                                           const std::string &functionName,
                                           const kernelInfo &info_){}

  template <>
  kernel_v* device_t<HSA>::loadKernelFromLibrary(const char *cache,
                                                 const std::string &functionName){}

  template <>
  memory_v* device_t<HSA>::wrapMemory(void *handle_,
                                      const uintptr_t bytes){}

  template <>
  memory_v* device_t<HSA>::wrapTexture(void *handle_,
                                       const int dim, const occa::dim &dims,
                                       occa::formatType type, const int permissions){}

  template <>
  memory_v* device_t<HSA>::malloc(const uintptr_t bytes,
                                  void *src){}

  template <>
  memory_v* device_t<HSA>::textureAlloc(const int dim, const occa::dim &dims,
                                        void *src,
                                        occa::formatType type, const int permissions){}

  template <>
  memory_v* device_t<HSA>::mappedAlloc(const uintptr_t bytes,
                                       void *src){}

  template <>
  void device_t<HSA>::free(){}

  template <>
  int device_t<HSA>::simdWidth(){}
  //==================================


  //---[ Error Handling ]-------------
  std::string hsaError(hsa_status_t s){
    switch(s){
    case HSA_STATUS_SUCCESS:                        return "HSA_STATUS_SUCCESS";
    case HSA_STATUS_INFO_BREAK:                     return "HSA_STATUS_INFO_BREAK";
    case HSA_STATUS_ERROR:                          return "HSA_STATUS_ERROR";
    case HSA_STATUS_ERROR_INVALID_ARGUMENT:         return "HSA_STATUS_ERROR_INVALID_ARGUMENT";
    case HSA_STATUS_ERROR_INVALID_QUEUE_CREATION:   return "HSA_STATUS_ERROR_INVALID_QUEUE_CREATION";
    case HSA_STATUS_ERROR_INVALID_ALLOCATION:       return "HSA_STATUS_ERROR_INVALID_ALLOCATION";
    case HSA_STATUS_ERROR_INVALID_AGENT:            return "HSA_STATUS_ERROR_INVALID_AGENT";
    case HSA_STATUS_ERROR_INVALID_REGION:           return "HSA_STATUS_ERROR_INVALID_REGION";
    case HSA_STATUS_ERROR_INVALID_SIGNAL:           return "HSA_STATUS_ERROR_INVALID_SIGNAL";
    case HSA_STATUS_ERROR_INVALID_QUEUE:            return "HSA_STATUS_ERROR_INVALID_QUEUE";
    case HSA_STATUS_ERROR_OUT_OF_RESOURCES:         return "HSA_STATUS_ERROR_OUT_OF_RESOURCES";
    case HSA_STATUS_ERROR_INVALID_PACKET_FORMAT:    return "HSA_STATUS_ERROR_INVALID_PACKET_FORMAT";
    case HSA_STATUS_ERROR_RESOURCE_FREE:            return "HSA_STATUS_ERROR_RESOURCE_FREE";
    case HSA_STATUS_ERROR_NOT_INITIALIZED:          return "HSA_STATUS_ERROR_NOT_INITIALIZED";
    case HSA_STATUS_ERROR_REFCOUNT_OVERFLOW:        return "HSA_STATUS_ERROR_REFCOUNT_OVERFLOW";
    case HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS:   return "HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS";
    case HSA_STATUS_ERROR_INVALID_INDEX:            return "HSA_STATUS_ERROR_INVALID_INDEX";
    case HSA_STATUS_ERROR_INVALID_ISA:              return "HSA_STATUS_ERROR_INVALID_ISA";
    case HSA_STATUS_ERROR_INVALID_ISA_NAME:         return "HSA_STATUS_ERROR_INVALID_ISA_NAME";
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT:      return "HSA_STATUS_ERROR_INVALID_CODE_OBJECT";
    case HSA_STATUS_ERROR_INVALID_EXECUTABLE:       return "HSA_STATUS_ERROR_INVALID_EXECUTABLE";
    case HSA_STATUS_ERROR_FROZEN_EXECUTABLE:        return "HSA_STATUS_ERROR_FROZEN_EXECUTABLE";
    case HSA_STATUS_ERROR_INVALID_SYMBOL_NAME:      return "HSA_STATUS_ERROR_INVALID_SYMBOL_NAME";
    case HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED: return "HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED";
    case HSA_STATUS_ERROR_VARIABLE_UNDEFINED:       return "HSA_STATUS_ERROR_VARIABLE_UNDEFINED";
    default:                                        return "UNKNOWN ERROR";
    };
  }
  //==================================
};

#endif
