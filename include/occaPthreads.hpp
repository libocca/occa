#ifndef OCCA_PTHREADS_HEADER
#define OCCA_PTHREADS_HEADER

#if (OCCA_OS & (LINUX_OS | OSX_OS))
#  if (OCCA_OS != WINUX_OS)
#    include <sys/sysctl.h>
#  endif
#  include <pthread.h>
#  include <dlfcn.h>
#else
#  include "vs/pthread.hpp"
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <fcntl.h>

#include <queue>

#include "occaBase.hpp"
#include "occaLibrary.hpp"

#include "occaKernelDefines.hpp"

namespace occa {
  //---[ Data Structs ]---------------
  struct PthreadKernelArg_t;
  typedef void (*PthreadLaunchHandle_t)(PthreadKernelArg_t &args);

  // [-] Hard-coded for now
  struct PthreadsDeviceData_t {
    int vendor;

    int coreCount;

    int pThreadCount;
    int schedule;

    pthread_t tid[OCCA_MAX_ARGS];

    int pendingJobs;

    std::queue<PthreadLaunchHandle_t> kernelLaunch[OCCA_MAX_ARGS];
    std::queue<PthreadKernelArg_t*> kernelArgs[OCCA_MAX_ARGS];

    pthread_mutex_t pendingJobsMutex, kernelMutex;
  };

  struct PthreadsKernelData_t {
    void *dlHandle, *handle;
    int pThreadCount;

    int *pendingJobs;

    std::queue<PthreadLaunchHandle_t> *kernelLaunch[50];
    std::queue<PthreadKernelArg_t*> *kernelArgs[50];

    pthread_mutex_t *pendingJobsMutex, *kernelMutex;
  };

  struct PthreadWorkerData_t {
    int rank, count;
    int pinnedCore;

    int *pendingJobs;

    std::queue<PthreadLaunchHandle_t> *kernelLaunch;
    std::queue<PthreadKernelArg_t*> *kernelArgs;

    pthread_mutex_t *pendingJobsMutex, *kernelMutex;
  };

  // [-] Hard-coded for now
  struct PthreadKernelArg_t {
    int rank, count;

    void *kernelHandle;

    int dims;
    occa::dim inner, outer;

    occa::kernelArg args[50];
  };

  static const int compact = (1 << 10);
  static const int scatter = (1 << 11);
  static const int manual  = (1 << 12);
  //==================================


  //---[ Kernel ]---------------------
  template <>
  kernel_t<Pthreads>::kernel_t();

  template <>
  kernel_t<Pthreads>::kernel_t(const kernel_t &k);

  template <>
  kernel_t<Pthreads>& kernel_t<Pthreads>::operator = (const kernel_t<Pthreads> &k);

  template <>
  kernel_t<Pthreads>::kernel_t(const kernel_t<Pthreads> &k);

  template <>
  std::string kernel_t<Pthreads>::fixBinaryName(const std::string &filename);

  template <>
  kernel_t<Pthreads>* kernel_t<Pthreads>::buildFromSource(const std::string &filename,
                                                          const std::string &functionName,
                                                          const kernelInfo &info_);

  template <>
  kernel_t<Pthreads>* kernel_t<Pthreads>::buildFromBinary(const std::string &filename,
                                                          const std::string &functionName);

  template <>
  kernel_t<Pthreads>* kernel_t<Pthreads>::loadFromLibrary(const char *cache,
                                                          const std::string &functionName);

  template <>
  uintptr_t kernel_t<Pthreads>::maximumInnerDimSize();

  template <>
  int kernel_t<Pthreads>::preferredDimSize();
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<Pthreads>::memory_t();

  template <>
  memory_t<Pthreads>::memory_t(const memory_t &m);

  template <>
  memory_t<Pthreads>& memory_t<Pthreads>::operator = (const memory_t &m);

  template <>
  void* memory_t<Pthreads>::getMemoryHandle();

  template <>
  void* memory_t<Pthreads>::getTextureHandle();

  template <>
  void memory_t<Pthreads>::copyFrom(const void *src,
                                    const uintptr_t bytes,
                                    const uintptr_t offset);

  template <>
  void memory_t<Pthreads>::copyFrom(const memory_v *src,
                                    const uintptr_t bytes,
                                    const uintptr_t destOffset,
                                    const uintptr_t srcOffset);

  template <>
  void memory_t<Pthreads>::copyTo(void *dest,
                                  const uintptr_t bytes,
                                  const uintptr_t destOffset);

  template <>
  void memory_t<Pthreads>::copyTo(memory_v *dest,
                                  const uintptr_t bytes,
                                  const uintptr_t srcOffset,
                                  const uintptr_t offset);

  template <>
  void memory_t<Pthreads>::asyncCopyFrom(const void *src,
                                         const uintptr_t bytes,
                                         const uintptr_t destOffset);

  template <>
  void memory_t<Pthreads>::asyncCopyFrom(const memory_v *src,
                                         const uintptr_t bytes,
                                         const uintptr_t srcOffset,
                                         const uintptr_t offset);

  template <>
  void memory_t<Pthreads>::asyncCopyTo(void *dest,
                                       const uintptr_t bytes,
                                       const uintptr_t offset);

  template <>
  void memory_t<Pthreads>::asyncCopyTo(memory_v *dest,
                                       const uintptr_t bytes,
                                       const uintptr_t destOffset,
                                       const uintptr_t srcOffset);

  template <>
  void memory_t<Pthreads>::mappedFree();

  template <>
  void memory_t<Pthreads>::free();
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<Pthreads>::device_t();

  template <>
  device_t<Pthreads>::device_t(const device_t<Pthreads> &k);

  template <>
  device_t<Pthreads>& device_t<Pthreads>::operator = (const device_t<Pthreads> &k);

  template <>
  void device_t<Pthreads>::setup(argInfoMap &aim);

  template <>
  void device_t<Pthreads>::addOccaHeadersToInfo(kernelInfo &info_);

  template <>
  std::string device_t<Pthreads>::getInfoSalt(const kernelInfo &info_);

  template <>
  deviceIdentifier device_t<Pthreads>::getIdentifier() const;

  template <>
  void device_t<Pthreads>::getEnvironmentVariables();

  template <>
  void device_t<Pthreads>::appendAvailableDevices(std::vector<device> &dList);

  template <>
  void device_t<Pthreads>::setCompiler(const std::string &compiler_);

  template <>
  void device_t<Pthreads>::setCompilerEnvScript(const std::string &compilerEnvScript_);

  template <>
  void device_t<Pthreads>::setCompilerFlags(const std::string &compilerFlags_);

  template <>
  void device_t<Pthreads>::flush();

  template <>
  void device_t<Pthreads>::finish();

  template <>
  void device_t<Pthreads>::waitFor(streamTag tag);

  template <>
  stream_t device_t<Pthreads>::createStream();

  template <>
  void device_t<Pthreads>::freeStream(stream_t s);

  template <>
  stream_t device_t<Pthreads>::wrapStream(void *handle_);

  template <>
  streamTag device_t<Pthreads>::tagStream();

  template <>
  double device_t<Pthreads>::timeBetween(const streamTag &startTag, const streamTag &endTag);

  template <>
  std::string device_t<Pthreads>::fixBinaryName(const std::string &filename);

  template <>
  kernel_v* device_t<Pthreads>::buildKernelFromSource(const std::string &filename,
                                                      const std::string &functionName,
                                                      const kernelInfo &info_);

  template <>
  kernel_v* device_t<Pthreads>::buildKernelFromBinary(const std::string &filename,
                                                      const std::string &functionName);

  template <>
  void device_t<Pthreads>::cacheKernelInLibrary(const std::string &filename,
                                                const std::string &functionName,
                                                const kernelInfo &info_);

  template <>
  kernel_v* device_t<Pthreads>::loadKernelFromLibrary(const char *cache,
                                                      const std::string &functionName);

  template <>
  void device_t<Pthreads>::free();

  template <>
  memory_v* device_t<Pthreads>::wrapMemory(void *handle_,
                                           const uintptr_t bytes);

  template <>
  memory_v* device_t<Pthreads>::wrapTexture(void *handle_,
                                            const int dim, const occa::dim &dims,
                                            occa::formatType type, const int permissions);

  template <>
  memory_v* device_t<Pthreads>::malloc(const uintptr_t bytes,
                                       void *src);

  template <>
  memory_v* device_t<Pthreads>::textureAlloc(const int dim, const occa::dim &dims,
                                             void *src,
                                             occa::formatType type, const int permissions);

  template <>
  memory_v* device_t<Pthreads>::mappedAlloc(const uintptr_t bytes,
                                            void *src);

  template <>
  int device_t<Pthreads>::simdWidth();
  //==================================

#include "operators/occaFunctionPointerTypeDefs.hpp"
#include "operators/occaPthreadsKernelOperators.hpp"

  //---[ Pthreads ]-------------------
  static void* pthreadLimbo(void *args){
    PthreadWorkerData_t &data = *((PthreadWorkerData_t*) args);

    // Thread affinity
#if (OCCA_OS == LINUX_OS) // Not WINUX
    cpu_set_t cpuHandle;
    CPU_ZERO(&cpuHandle);
    CPU_SET(data.pinnedCore, &cpuHandle);
#else
    // NBN: affinity on hyperthreaded multi-socket systems?
    if(data.rank == 0)
      fprintf(stderr, "[Pthreads] Affinity not guaranteed in this OS\n");
    // BOOL SetProcessAffinityMask(HANDLE hProcess,DWORD_PTR dwProcessAffinityMask);
#endif

    while(true){
      // Fence local data (incase of out-of-socket updates)
#if (OCCA_OS & (LINUX_OS | OSX_OS))
      __asm__ __volatile__ ("lfence");
#else
      __faststorefence(); // NBN: x64 only?
#endif

      if( *(data.pendingJobs) ){
        pthread_mutex_lock(data.kernelMutex);

        std::cout.flush();

        PthreadLaunchHandle_t launchKernel = data.kernelLaunch->front();
        data.kernelLaunch->pop();

        PthreadKernelArg_t &launchArgs = *(data.kernelArgs->front());
        data.kernelArgs->pop();

        pthread_mutex_unlock(data.kernelMutex);

        launchKernel(launchArgs);

        //---[ Barrier ]----------------
        pthread_mutex_lock(data.pendingJobsMutex);
        --( *(data.pendingJobs) );
        pthread_mutex_unlock(data.pendingJobsMutex);

        while((*data.pendingJobs) % data.count){
#if (OCCA_OS & (LINUX_OS | OSX_OS))
          __asm__ __volatile__ ("lfence");
#else
          __faststorefence(); // NBN: x64 only?
#endif
        }
        //==============================
      }
    }

    return NULL;
  }
  //==================================
};

#endif
