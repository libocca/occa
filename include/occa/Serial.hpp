#ifndef OCCA_SERIAL_HEADER
#define OCCA_SERIAL_HEADER

#if (OCCA_OS & (LINUX_OS | OSX_OS))
#  if   (OCCA_OS == LINUX_OS)
#    include <sys/sysinfo.h>
#  elif (OCCA_OS == OSX_OS)
#    include <mach/mach.h>
#    include <mach/mach_host.h>
#  endif
#  if (OCCA_OS != WINUX_OS)
#    include <sys/sysctl.h>
#  endif
#  include <sys/wait.h>
#  include <dlfcn.h>
#else
#  include <windows.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <fcntl.h>

#include "occa/base.hpp"
#include "occa/library.hpp"

namespace occa {
  //---[ Data Structs ]---------------
  struct SerialKernelData_t {
    void *dlHandle;
    handleFunction_t handle;

    void *vArgs[2*OCCA_MAX_ARGS];
  };

  struct SerialDeviceData_t {
    int vendor;
  };
  //==================================


  //---[ Helper Functions ]-----------
  namespace cpu {
    namespace vendor {
      static const int notFound     = 0;

      static const int b_GNU          = 0;
      static const int b_LLVM         = 1;
      static const int b_Intel        = 2;
      static const int b_Pathscale    = 3;
      static const int b_IBM          = 4;
      static const int b_PGI          = 5;
      static const int b_HP           = 6;
      static const int b_VisualStudio = 7;
      static const int b_Cray         = 8;
      static const int b_max          = 9;

      static const int GNU          = (1 << b_GNU);          // gcc    , g++
      static const int LLVM         = (1 << b_LLVM);         // clang  , clang++
      static const int Intel        = (1 << b_Intel);        // icc    , icpc
      static const int Pathscale    = (1 << b_Pathscale);    // pathCC
      static const int IBM          = (1 << b_IBM);          // xlc    , xlc++
      static const int PGI          = (1 << b_PGI);          // pgcc   , pgc++
      static const int HP           = (1 << b_HP);           // aCC
      static const int VisualStudio = (1 << b_VisualStudio); // cl.exe
      static const int Cray         = (1 << b_Cray);         // cc     , CC
    }

    std::string getFieldFrom(const std::string &command,
                             const std::string &field);

    std::string getProcessorName();
    int getCoreCount();
    int getProcessorFrequency();
    std::string getProcessorCacheSize(int level);
    uintptr_t installedRAM();
    uintptr_t availableRAM();

    std::string getDeviceListInfo();

    int compilerVendor(const std::string &compiler);

    std::string compilerSharedBinaryFlags(const std::string &compiler);
    std::string compilerSharedBinaryFlags(const int vendor_);

    void addSharedBinaryFlagsTo(const std::string &compiler, std::string &flags);
    void addSharedBinaryFlagsTo(const int vendor_, std::string &flags);

    void* malloc(uintptr_t bytes);
    void free(void *ptr);

    void* dlopen(const std::string &filename,
                 const std::string &hash = "");

    handleFunction_t dlsym(void *dlHandle,
                           const std::string &functionName,
                           const std::string &hash = "");

    void runFunction(handleFunction_t f,
                     const int *occaKernelInfoArgs,
                     int occaInnerId0, int occaInnerId1, int occaInnerId2,
                     int argc, void **args);
  }
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
  void* kernel_t<Serial>::getKernelHandle();

  template <>
  void* kernel_t<Serial>::getProgramHandle();

  template <>
  std::string kernel_t<Serial>::fixBinaryName(const std::string &filename);

  template <>
  kernel_t<Serial>* kernel_t<Serial>::buildFromSource(const std::string &filename,
                                                      const std::string &functionName,
                                                      const kernelInfo &info_);

  template <>
  kernel_t<Serial>* kernel_t<Serial>::buildFromBinary(const std::string &filename,
                                                      const std::string &functionName);

  template <>
  kernel_t<Serial>* kernel_t<Serial>::loadFromLibrary(const char *cache,
                                                      const std::string &functionName);

  template <>
  uintptr_t kernel_t<Serial>::maximumInnerDimSize();

  template <>
  int kernel_t<Serial>::preferredDimSize();

  template <>
  void kernel_t<Serial>::runFromArguments(const int kArgc, const kernelArg *kArgs);

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
  void* device_t<Serial>::getContextHandle();

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
  void device_t<Serial>::flush();

  template <>
  void device_t<Serial>::finish();

  template <>
  void device_t<Serial>::waitFor(streamTag tag);

  template <>
  stream_t device_t<Serial>::createStream();

  template <>
  void device_t<Serial>::freeStream(stream_t s);

  template <>
  stream_t device_t<Serial>::wrapStream(void *handle_);

  template <>
  streamTag device_t<Serial>::tagStream();

  template <>
  double device_t<Serial>::timeBetween(const streamTag &startTag, const streamTag &endTag);

  template <>
  std::string device_t<Serial>::fixBinaryName(const std::string &filename);

  template <>
  kernel_v* device_t<Serial>::buildKernelFromSource(const std::string &filename,
                                                    const std::string &functionName,
                                                    const kernelInfo &info_);

  template <>
  kernel_v* device_t<Serial>::buildKernelFromBinary(const std::string &filename,
                                                    const std::string &functionName);

  template <>
  void device_t<Serial>::cacheKernelInLibrary(const std::string &filename,
                                              const std::string &functionName,
                                              const kernelInfo &info_);

  template <>
  kernel_v* device_t<Serial>::loadKernelFromLibrary(const char *cache,
                                                    const std::string &functionName);

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
  uintptr_t device_t<Serial>::memorySize();

  template <>
  void device_t<Serial>::free();

  template <>
  int device_t<Serial>::simdWidth();
  //==================================
}

#endif
