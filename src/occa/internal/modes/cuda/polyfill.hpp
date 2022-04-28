#include <occa/defines.hpp>

#ifndef OCCA_INTERNAL_MODES_CUDA_POLYFILL_HEADER
#define OCCA_INTERNAL_MODES_CUDA_POLYFILL_HEADER

#if OCCA_CUDA_ENABLED
#  include <cuda.h>
#else

#define CUDA_VERSION 0

// Wrap in the occa namespace so as long as we don't use ::CUmodule, the two
//   - CUmodule
//   - occa::CUmodule
// are indistinguisable inside the occa namespace
namespace occa {
  //---[ Types ]------------------------
  struct _CUdeviceptr {};
  typedef struct _CUcontext*            CUcontext;
  typedef int                           CUdevice;
  typedef struct _CUdeviceptr*          CUdeviceptr;
  typedef struct _CUevent*              CUevent;
  typedef struct _CUfunction*           CUfunction;
  typedef struct _CUfunction_attribute* CUfunction_attribute;
  typedef struct _CUmodule*             CUmodule;
  typedef struct _CUstream*             CUstream;

  //---[ Enums ]------------------------
  static const int CU_DEVICE_CPU = 0;
  static const int CU_EVENT_DEFAULT = 0;
  static const int CU_MEM_ATTACH_GLOBAL = 0;
  static const int CU_MEM_ATTACH_HOST = 0;
  static const int CU_STREAM_DEFAULT = 0;
  static const int CU_STREAM_NON_BLOCKING = 0;
  static const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 0;
  static const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 0;

  static const CUfunction_attribute CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = NULL;

  enum CUresult {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE,
    CUDA_ERROR_OUT_OF_MEMORY,
    CUDA_ERROR_NOT_INITIALIZED,
    CUDA_ERROR_DEINITIALIZED,
    CUDA_ERROR_PROFILER_DISABLED,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED,
    CUDA_ERROR_PROFILER_ALREADY_STARTED,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED,
    CUDA_ERROR_NO_DEVICE,
    CUDA_ERROR_INVALID_DEVICE,
    CUDA_ERROR_INVALID_IMAGE,
    CUDA_ERROR_INVALID_CONTEXT,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
    CUDA_ERROR_MAP_FAILED,
    CUDA_ERROR_UNMAP_FAILED,
    CUDA_ERROR_ARRAY_IS_MAPPED,
    CUDA_ERROR_ALREADY_MAPPED,
    CUDA_ERROR_NO_BINARY_FOR_GPU,
    CUDA_ERROR_ALREADY_ACQUIRED,
    CUDA_ERROR_NOT_MAPPED,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER,
    CUDA_ERROR_ECC_UNCORRECTABLE,
    CUDA_ERROR_UNSUPPORTED_LIMIT,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
    CUDA_ERROR_INVALID_SOURCE,
    CUDA_ERROR_FILE_NOT_FOUND,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
    CUDA_ERROR_OPERATING_SYSTEM,
    CUDA_ERROR_INVALID_HANDLE,
    CUDA_ERROR_NOT_FOUND,
    CUDA_ERROR_NOT_READY,
    CUDA_ERROR_LAUNCH_FAILED,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
    CUDA_ERROR_LAUNCH_TIMEOUT,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE,
    CUDA_ERROR_CONTEXT_IS_DESTROYED,
    CUDA_ERROR_ASSERT,
    CUDA_ERROR_TOO_MANY_PEERS,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
    CUDA_ERROR_NOT_PERMITTED,
    CUDA_ERROR_NOT_SUPPORTED,
    CUDA_ERROR_INVALID_PTX,
    CUDA_ERROR_ILLEGAL_ADDRESS,
    CUDA_ERROR_HARDWARE_STACK_ERROR,
    CUDA_ERROR_ILLEGAL_INSTRUCTION,
    CUDA_ERROR_MISALIGNED_ADDRESS,
    CUDA_ERROR_INVALID_ADDRESS_SPACE,
    CUDA_ERROR_INVALID_PC,
    CUDA_ERROR_NVLINK_UNCORRECTABLE,
    OCCA_CUDA_IS_NOT_ENABLED
  };

  //---[ Methods ]----------------------
  inline CUresult cuInit(unsigned int Flags) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuLaunchKernel(CUfunction f,
                                 unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                 unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                 unsigned int sharedMemBytes,
                                 CUstream hStream,
                                 void **kernelParams,
                                 void **extra) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  //   ---[ Context ]-------------------
  inline CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuCtxSetCurrent(CUcontext ctx) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  //   ---[ Device ]--------------------
  inline CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
    // [Deprecated]
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuDeviceGetCount(int *count) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  //   ---[ Event ]---------------------
  inline CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuEventDestroy(CUevent hEvent) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuEventSynchronize(CUevent hEvent) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }


  //   ---[ Memory ]--------------------
  inline CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuMemAllocHost(void **pp, size_t bytesize) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuMemFree(CUdeviceptr dptr) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuMemFreeHost(void *p) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuMemHostGetDevicePointer(CUdeviceptr *dptr, void *p, unsigned int Flags) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuMemPrefetchAsync(CUdeviceptr *devPtr, size_t count, CUdevice dstDevice, CUstream hStream) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, const CUdeviceptr srcDevice, size_t ByteCount) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, const CUdeviceptr srcDevice, size_t ByteCount, CUstream hstream) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuMemcpyDtoH(void *dstHost, const CUdeviceptr srcDevice, size_t ByteCount) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuMemcpyDtoHAsync(void *dstHost, const CUdeviceptr srcDevice, size_t ByteCount, CUstream hstream) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hstream) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                               CUdeviceptr srcDevice, CUcontext srcContext,
                               size_t ByteCount) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                                    CUdeviceptr srcDevice, CUcontext srcContext,
                                    size_t ByteCount, CUstream hStream) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  // Version-dependent methods:
  // inline CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advice advice, CUdevice device)

  //   ---[ Module ]--------------------
  inline CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuModuleLoad(CUmodule *module, const char *fname) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuModuleUnload(CUmodule hmod) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  //   ---[ Stream ]--------------------
  inline CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuStreamDestroy(CUstream hStream) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }

  inline CUresult cuStreamSynchronize(CUstream hStream) {
    return OCCA_CUDA_IS_NOT_ENABLED;
  }
}

#endif
#endif
