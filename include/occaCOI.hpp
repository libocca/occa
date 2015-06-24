#if OCCA_COI_ENABLED
#  ifndef OCCA_COI_HEADER
#  define OCCA_COI_HEADER

#include "occaBase.hpp"
#include "occaLibrary.hpp"

#include "occaKernelDefines.hpp"

#include "occaDefines.hpp"

#include <intel-coi/source/COIProcess_source.h>
#include <intel-coi/source/COIEngine_source.h>
#include <intel-coi/source/COIPipeline_source.h>
#include <intel-coi/source/COIEvent_source.h>
#include <intel-coi/source/COIBuffer_source.h>

namespace occa {
  typedef COIENGINE   coiDevice;
  typedef COIPROCESS  coiChief;   // Ruler of all threads

  typedef COIEVENT    coiEvent;

  typedef COIFUNCTION      coiKernel;
  typedef COIBUFFER        coiMemory;
  typedef COI_ACCESS_FLAGS coiMemoryFlags;

  typedef COIRESULT coiStatus;

  //---[ Data Structs ]---------------
  struct coiStream {
    COIPIPELINE handle;
    coiEvent lastEvent;
  };

  struct COIKernelData_t {
    coiChief chiefID;
    coiKernel kernel;

    // [-] Hard-coded for now
    coiMemory deviceArgv[50];
    coiMemoryFlags deviceFlags[50];

    char hostArgv[100 + 6*4 + 50*(4 + 8)];
    //            ^     ^      ^  ^   ^__[Max Bytes]
    // [Padding]__|     |      |  |__[Type]
    //                  |      |__[Maximum Args]
    //                  |__[KernelArgs]
  };

  struct COIDeviceData_t {
    coiDevice deviceID;
    coiChief chiefID;

    coiKernel kernelWrapper[50];
  };
  //==================================


  //---[ Helper Functions ]-----------
  namespace coi {
    void initDevice(COIENGINE &device, COIPROCESS &chief);

    occa::device wrapDevice(COIENGINE device);
  };
  //==================================


  //---[ Kernel ]---------------------
  template <>
  kernel_t<COI>::kernel_t();

  template <>
  kernel_t<COI>::kernel_t(const kernel_t &k);

  template <>
  kernel_t<COI>& kernel_t<COI>::operator = (const kernel_t<COI> &k);

  template <>
  kernel_t<COI>::kernel_t(const kernel_t<COI> &k);

  template <>
  std::string &filename);

  template <>
  kernel_t<COI>* kernel_t<COI>::buildFromSource(const std::string &filename,
                                                const std::string &functionName,
                                                const kernelInfo &info_);

  template <>
  kernel_t<COI>* kernel_t<COI>::buildFromBinary(const std::string &filename,
                                                const std::string &functionName);

  template <>
  kernel_t<COI>* kernel_t<COI>::loadFromLibrary(const char *cache,
                                                const std::string &functionName);

  template <>
  uintptr_t kernel_t<COI>::maximumInnerDimSize();

  template <>
  int kernel_t<COI>::preferredDimSize();

  template <>
  void kernel_t<COI>::free();
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<COI>::memory_t();

  template <>
  memory_t<COI>::memory_t(const memory_t &m);

  template <>
  memory_t<COI>& memory_t<COI>::operator = (const memory_t &m);

  template <>
  void* memory_t<COI>::getMemoryHandle();

  template <>
  void* memory_t<COI>::getTextureHandle();

  template <>
  void memory_t<COI>::copyFrom(const void *src,
                               const uintptr_t bytes,
                               const uintptr_t offset);

  template <>
  void memory_t<COI>::copyFrom(const memory_v *src,
                               const uintptr_t bytes,
                               const uintptr_t destOffset,
                               const uintptr_t srcOffset);

  template <>
  void memory_t<COI>::copyTo(void *dest,
                             const uintptr_t bytes,
                             const uintptr_t offset);

  template <>
  void memory_t<COI>::copyTo(memory_v *dest,
                             const uintptr_t bytes,
                             const uintptr_t destOffset,
                             const uintptr_t srcOffset);

  template <>
  void memory_t<COI>::asyncCopyFrom(const void *src,
                                    const uintptr_t bytes,
                                    const uintptr_t destOffset);

  template <>
  void memory_t<COI>::asyncCopyFrom(const memory_v *src,
                                    const uintptr_t bytes,
                                    const uintptr_t srcOffset,
                                    const uintptr_t offset);

  template <>
  void memory_t<COI>::asyncCopyTo(void *dest,
                                  const uintptr_t bytes,
                                  const uintptr_t offset);

  template <>
  void memory_t<COI>::asyncCopyTo(memory_v *dest,
                                  const uintptr_t bytes,
                                  const uintptr_t destOffset,
                                  const uintptr_t srcOffset);

  template <>
  void memory_t<COI>::mappedFree();

  template <>
  void memory_t<COI>::free();
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<COI>::device_t();

  template <>
  device_t<COI>::device_t(const device_t<COI> &k);

  template <>
  device_t<COI>& device_t<COI>::operator = (const device_t<COI> &k);

  template <>
  void device_t<COI>::setup(argInfoMap &aim);

  template <>
  void device_t<COI>::addOccaHeadersToInfo(kernelInfo &info_);

  template <>
  std::string device_t<COI>::getInfoSalt(const kernelInfo &info_);

  template <>
  deviceIdentifier device_t<COI>::getIdentifier() const;

  template <>
  void device_t<COI>::getEnvironmentVariables();

  template <>
  void device_t<COI>::appendAvailableDevices(std::vector<device> &dList);

  template <>
  void device_t<COI>::setCompiler(const std::string &compiler_);

  template <>
  void device_t<COI>::setCompilerEnvScript(const std::string &compilerEnvScript_);

  template <>
  void device_t<COI>::setCompilerFlags(const std::string &compilerFlags_);

  template <>
  void device_t<COI>::flush();

  template <>
  void device_t<COI>::finish();

  template <>
  void device_t<COI>::waitFor(streamTag tag);

  template <>
  stream_t device_t<COI>::createStream();

  template <>
  void device_t<COI>::freeStream(stream_t s);

  template <>
  stream_t device_t<COI>::wrapStream(void *handle_);

  template <>
  streamTag device_t<COI>::tagStream();

  template <>
  double device_t<COI>::timeBetween(const streamTag &startTag, const streamTag &endTag);

  template <>
  kernel_v* device_t<COI>::buildKernelFromSource(const std::string &filename,
                                                 const std::string &functionName,
                                                 const kernelInfo &info_);

  template <>
  kernel_v* device_t<COI>::buildKernelFromBinary(const std::string &filename,
                                                 const std::string &functionName);

  template <>
  void device_t<COI>::cacheKernelInLibrary(const std::string &filename,
                                           const std::string &functionName,
                                           const kernelInfo &info_);

  template <>
  kernel_v* device_t<COI>::loadKernelFromLibrary(const char *cache,
                                                 const std::string &functionName);

  template <>
  memory_v* device_t<COI>::wrapMemory(void *handle_,
                                      const uintptr_t bytes);

  template <>
  memory_v* device_t<COI>::wrapTexture(void *handle_,
                                       const int dim, const occa::dim &dims,
                                       occa::formatType type, const int permissions);

  template <>
  memory_v* device_t<COI>::malloc(const uintptr_t bytes,
                                  void *src);

  template <>
  memory_v* device_t<COI>::textureAlloc(const int dim, const occa::dim &dims,
                                        void *src,
                                        occa::formatType type, const int permissions);

  template <>
  memory_v* device_t<COI>::mappedAlloc(const uintptr_t bytes,
                                       void *src);

  template <>
  void device_t<COI>::free();

  template <>
  int device_t<COI>::simdWidth();
  //==================================

#include "operators/occaCOIFunctionPointerTypeDefs.hpp"
#include "operators/occaCOIKernelOperators.hpp"

  //---[ Error Handling ]-------------
  std::string coiError(coiStatus e);
  //==================================
};

#  endif
#endif
