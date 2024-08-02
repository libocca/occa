#include <occa/defines.hpp>

#ifndef OCCA_INTERNAL_MODES_HIP_POLYFILL_HEADER
#define OCCA_INTERNAL_MODES_HIP_POLYFILL_HEADER

#if OCCA_HIP_ENABLED
#  include <hip/hip_runtime_api.h>
#else

#include <sys/types.h>
#undef   minor
#undef   major

// Wrap in the occa namespace so as long as we don't use ::hipModule_t, the two
//   - hipModule_t
//   - occa::hipModule_t
// are indistinguisable inside the occa namespace
namespace occa {
  //---[ Types ]------------------------
  typedef struct _hipCtx_t*               hipCtx_t;
  typedef int                             hipDevice_t;
  typedef void*                           hipDeviceptr_t;
  typedef struct _hipEvent_t*             hipEvent_t;
  typedef struct _hipFunction_t*          hipFunction_t;
  typedef struct _hipFunctionAttribute_t* hipFunctionAttribute_t;
  typedef struct _hipModule_t*            hipModule_t;
  typedef struct _hipStream_t*            hipStream_t;

  //---[ Enums ]------------------------
  static const int HIP_LAUNCH_PARAM_BUFFER_POINTER = 0;
  static const int HIP_LAUNCH_PARAM_BUFFER_SIZE = 0;
  static const int HIP_LAUNCH_PARAM_END = 0;
  static const int hipStreamNonBlocking = 0;

  class hipDeviceProp_t {
   public:
    //INFO: original HIP has exact this definition of name field
    char name[256];
    size_t totalGlobalMem;
    int maxThreadsPerBlock;
    char gcnArchName[256];
    int major;
    int minor;

    inline hipDeviceProp_t() :
        name{0},
        totalGlobalMem(0),
        maxThreadsPerBlock(-1),
        major(-1),
        minor(-1) {}
  };

  enum hipError_t {
    hipSuccess = 0,
    hipErrorInvalidValue,
    hipErrorMemoryAllocation,
    hipErrorNotInitialized,
    hipErrorDeinitialized,
    hipErrorProfilerDisabled,
    hipErrorProfilerNotInitialized,
    hipErrorProfilerAlreadyStarted,
    hipErrorProfilerAlreadyStopped,
    hipErrorNoDevice,
    hipErrorInvalidDevice,
    hipErrorInvalidImage,
    hipErrorInvalidContext,
    hipErrorContextAlreadyCurrent,
    hipErrorMapFailed,
    hipErrorUnmapFailed,
    hipErrorArrayIsMapped,
    hipErrorAlreadyMapped,
    hipErrorNoBinaryForGpu,
    hipErrorAlreadyAcquired,
    hipErrorNotMapped,
    hipErrorNotMappedAsArray,
    hipErrorNotMappedAsPointer,
    hipErrorECCNotCorrectable,
    hipErrorUnsupportedLimit,
    hipErrorContextAlreadyInUse,
    hipErrorPeerAccessUnsupported,
    hipErrorInvalidSource,
    hipErrorFileNotFound,
    hipErrorSharedObjectSymbolNotFound,
    hipErrorSharedObjectInitFailed,
    hipErrorOperatingSystem,
    hipErrorInvalidHandle,
    hipErrorNotFound,
    hipErrorNotReady,
    hipErrorLaunchOutOfResources,
    hipErrorLaunchTimeOut,
    hipErrorPeerAccessAlreadyEnabled,
    hipErrorPeerAccessNotEnabled,
    hipErrorHostMemoryAlreadyRegistered,
    hipErrorHostMemoryNotRegistered,
    OCCA_HIP_IS_NOT_ENABLED
  };

  //---[ Methods ]----------------------
  inline hipError_t hipInit(unsigned int Flags) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipDriverGetVersion(int *pVersion) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipFuncGetAttribute(int *pi, hipFunctionAttribute_t attrib, hipFunction_t hfunc) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipModuleLaunchKernel(hipFunction_t f,
                                          unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                          unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                          unsigned int sharedMemBytes,
                                          hipStream_t hStream,
                                          void **kernelParams,
                                          void **extra) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  //   ---[ Context ]-------------------
  inline hipError_t hipCtxCreate(hipCtx_t *pctx, unsigned int flags, hipDevice_t dev) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipCtxDestroy(hipCtx_t ctx) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipCtxEnablePeerAccess(hipCtx_t peerContext, unsigned int Flags) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  //   ---[ Device ]--------------------
  inline hipError_t hipDeviceCanAccessPeer(int *canAccessPeer, hipDevice_t dev, hipDevice_t peerDev) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipDeviceComputeCapability(int *major, int *minor, hipDevice_t dev) {
    // [Deprecated]
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipDeviceGet(hipDevice_t *device, int ordinal) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipGetDeviceProperties(hipDeviceProp_t *deviceProps, hipDevice_t device) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipGetDeviceCount(int *count) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipDeviceGetName(char *name, int len, hipDevice_t dev) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipSetDevice(hipDevice_t device) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipDeviceTotalMem(size_t *bytes, hipDevice_t dev) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  //   ---[ Event ]---------------------
  inline hipError_t hipEventCreate(hipEvent_t *phEvent) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipEventDestroy(hipEvent_t hEvent) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipEventElapsedTime(float *pMilliseconds, hipEvent_t hStart, hipEvent_t hEnd) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipEventRecord(hipEvent_t hEvent, hipStream_t hStream) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipEventSynchronize(hipEvent_t hEvent) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }


  //   ---[ Memory ]--------------------
  inline hipError_t hipMalloc(hipDeviceptr_t *dptr, size_t bytesize) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipHostMalloc(void **pp, size_t bytesize) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipMemAllocManaged(hipDeviceptr_t *dptr, size_t bytesize, unsigned int flags) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipFree(hipDeviceptr_t dptr) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipHostFree(void *p) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipHostGetDevicePointer(hipDeviceptr_t *dptr, void *p, unsigned int Flags) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipMemPrefetchAsync(hipDeviceptr_t *dptr, size_t count, hipDevice_t dstDevice, hipStream_t hStream) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipMemcpyDtoD(hipDeviceptr_t dstDevice, const hipDeviceptr_t srcDevice, size_t ByteCount) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dstDevice, const hipDeviceptr_t srcDevice, size_t ByteCount, hipStream_t hstream) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipMemcpyDtoH(void *dstHost, const hipDeviceptr_t srcDevice, size_t ByteCount) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipMemcpyDtoHAsync(void *dstHost, const hipDeviceptr_t srcDevice, size_t ByteCount, hipStream_t hstream) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipMemcpyHtoD(hipDeviceptr_t dstDevice, const void *srcHost, size_t ByteCount) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dstDevice, const void *srcHost, size_t ByteCount, hipStream_t hstream) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipMemcpyPeer(hipDeviceptr_t dstDevice, hipCtx_t dstContext,
                                  hipDeviceptr_t srcDevice, hipCtx_t srcContext,
                                  size_t ByteCount) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipMemcpyPeerAsync(hipDeviceptr_t dstDevice, hipCtx_t dstContext,
                                       hipDeviceptr_t srcDevice, hipCtx_t srcContext,
                                       size_t ByteCount, hipStream_t hStream) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  //   ---[ Module ]--------------------
  inline hipError_t hipModuleGetFunction(hipFunction_t *hfunc, hipModule_t hmod, const char *name) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipModuleLoad(hipModule_t *module, const char *fname) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipModuleUnload(hipModule_t hmod) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  //   ---[ Stream ]--------------------
  inline hipError_t hipStreamCreate(hipStream_t *phStream) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipStreamCreateWithFlags(hipStream_t *phStream, unsigned int flags) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipStreamDestroy(hipStream_t hStream) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipStreamSynchronize(hipStream_t hStream) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }

  inline hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
    return OCCA_HIP_IS_NOT_ENABLED;
  }
}

#endif
#endif
