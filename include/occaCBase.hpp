#ifndef OCCA_CBASE_HEADER
#define OCCA_CBASE_HEADER

#include "ocl_preprocessor.hpp"

#include "stdlib.h"

#if OCCA_OPENCL_ENABLED
#  if   OCCA_OS == LINUX_OS
#    include <CL/cl.h>
#    include <CL/cl_gl.h>
#  elif OCCA_OS == OSX_OS
#    include <OpenCL/OpenCl.h>
#  endif
#endif

#if OCCA_CUDA_ENABLED
#  include <cuda.h>
#endif

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
#  define LIBOCCA_CALLINGCONV
#  define LIBOCCA_API
#else
#  define LIBOCCA_CALLINGCONV __stdcall
#  ifdef LIBOCCA_C_EXPORTS
//#define LIBOCCA_API __declspec(dllexport)
#    define LIBOCCA_API
#  else
//#define LIBOCCA_API __declspec(dllimport)
#    define LIBOCCA_API
#  endif
#endif

#define OCCA_TYPE_MEMORY 0
#define OCCA_TYPE_INT    1
#define OCCA_TYPE_UINT   2
#define OCCA_TYPE_CHAR   3
#define OCCA_TYPE_UCHAR  4
#define OCCA_TYPE_SHORT  5
#define OCCA_TYPE_USHORT 6
#define OCCA_TYPE_LONG   7
#define OCCA_TYPE_ULONG  8
#define OCCA_TYPE_FLOAT  9
#define OCCA_TYPE_DOUBLE 10
#define OCCA_TYPE_STRING 11
#define OCCA_TYPE_COUNT  12

#define OCCA_ARG_COUNT(...) OCCA_ARG_COUNT2(__VA_ARGS__, 25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1)
#define OCCA_ARG_COUNT2(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,_17,_18,_19,_20,_21,_22,_23,_24,_25, N, ...) N

#define occaKernelRun(...) OCCA_C_RUN_KERNEL1( OCL_SUB(OCCA_ARG_COUNT(__VA_ARGS__), 1) , __VA_ARGS__)
#define OCCA_C_RUN_KERNEL1(...) OCCA_C_RUN_KERNEL2(__VA_ARGS__)
#define OCCA_C_RUN_KERNEL2(...) OCCA_C_RUN_KERNEL3(__VA_ARGS__)
#define OCCA_C_RUN_KERNEL3(N, kernel, ...) occaKernelRun##N(kernel, __VA_ARGS__)

//---[ Declarations ]-------------------

#define OCCA_C_KERNEL_RUN_DECLARATION_ARGS(N) , void *arg##N
#define OCCA_C_KERNEL_RUN_DECLARATION(N)                                \
  LIBOCCA_API void LIBOCCA_CALLINGCONV occaKernelRun##N(occaKernel kernel OCL_FOR(1, N, OCCA_C_KERNEL_RUN_DECLARATION_ARGS));

#define OCCA_C_KERNEL_RUN_DECLARATIONS                            \
  OCL_FOR_2(1, OCL_MAX_FOR_LOOPS, OCCA_C_KERNEL_RUN_DECLARATION)

//---[ Definitions ]--------------------

#define OCCA_C_KERNEL_RUN_ADD_ARG(N)                                    \
  {                                                                     \
    occaMemory_t &__occa_memory__ = *((occaMemory_t*) arg##N);          \
                                                                        \
    if(__occa_memory__.type == OCCA_TYPE_MEMORY){                       \
      __occa_kernel__.addArgument(N - 1, occa::kernelArg(__occa_memory__.mem)); \
    }                                                                   \
    else{                                                               \
      occaType_t &__occa_type__ = *((occaType_t*) arg##N);              \
                                                                        \
      __occa_kernel__.addArgument(N - 1,                                \
                                  occa::kernelArg(__occa_type__.value,  \
                                                  occaTypeSize[__occa_type__.type], \
                                                  false));              \
    }                                                                   \
  }

#define OCCA_C_KERNEL_RUN_DEFINITION(N)                                 \
  void LIBOCCA_CALLINGCONV occaKernelRun##N(occaKernel kernel OCL_FOR(1, N, OCCA_C_KERNEL_RUN_DECLARATION_ARGS)){ \
    occa::kernel &__occa_kernel__  = *((occa::kernel*) kernel);         \
    __occa_kernel__.clearArgumentList();                                \
                                                                        \
    OCL_FOR(1, N, OCCA_C_KERNEL_RUN_ADD_ARG);                           \
                                                                        \
    __occa_kernel__.runFromArguments();                                 \
  }

#define OCCA_C_KERNEL_RUN_DEFINITIONS                           \
  OCL_FOR_2(1, OCL_MAX_FOR_LOOPS, OCCA_C_KERNEL_RUN_DEFINITION)

#  ifdef __cplusplus
extern "C" {
#  endif

  typedef void* occaDevice;
  typedef void* occaKernel;

  typedef struct occaMemory_t* occaMemory;

  typedef struct occaType_t*         occaType;
  typedef struct occaArgumentList_t* occaArgumentList;

  typedef void* occaStream;

  typedef union occaTag_t {
    double tagTime;
    void* otherStuff;
  } occaTag;

  typedef void* occaKernelInfo;

  typedef struct occaDim_t {
    uintptr_t x, y, z;
  } occaDim;

  extern LIBOCCA_API occaKernelInfo occaNoKernelInfo;

  extern LIBOCCA_API const uintptr_t occaAutoSize;
  extern LIBOCCA_API const uintptr_t occaNoOffset;

  extern LIBOCCA_API const uintptr_t occaTypeSize[OCCA_TYPE_COUNT];

  //---[ TypeCasting ]------------------
  LIBOCCA_API occaType LIBOCCA_CALLINGCONV occaInt(int value);
  LIBOCCA_API occaType LIBOCCA_CALLINGCONV occaUInt(unsigned int value);

  LIBOCCA_API occaType LIBOCCA_CALLINGCONV occaChar(char value);
  LIBOCCA_API occaType LIBOCCA_CALLINGCONV occaUChar(unsigned char value);

  LIBOCCA_API occaType LIBOCCA_CALLINGCONV occaShort(short value);
  LIBOCCA_API occaType LIBOCCA_CALLINGCONV occaUShort(unsigned short value);

  LIBOCCA_API occaType LIBOCCA_CALLINGCONV occaLong(long value);
  LIBOCCA_API occaType LIBOCCA_CALLINGCONV occaULong(unsigned long value);

  LIBOCCA_API occaType LIBOCCA_CALLINGCONV occaFloat(float value);
  LIBOCCA_API occaType LIBOCCA_CALLINGCONV occaDouble(double value);

  LIBOCCA_API occaType LIBOCCA_CALLINGCONV occaString(char *str);
  //====================================


  //---[ Device ]-----------------------
  LIBOCCA_API const char* LIBOCCA_CALLINGCONV occaDeviceMode(occaDevice device);

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaDeviceSetCompiler(occaDevice device,
                                                             const char *compiler);

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaDeviceSetCompilerFlags(occaDevice device,
                                                                  const char *compilerFlags);

  LIBOCCA_API occaDevice LIBOCCA_CALLINGCONV occaGetDevice(const char *infos);

  LIBOCCA_API occaDevice LIBOCCA_CALLINGCONV occaGetDeviceFromArgs(const char *mode,
                                                                   int arg1, int arg2);

  LIBOCCA_API occaKernel LIBOCCA_CALLINGCONV occaBuildKernelFromSource(occaDevice device,
                                                                       const char *filename,
                                                                       const char *functionName,
                                                                       occaKernelInfo info);

  LIBOCCA_API occaKernel LIBOCCA_CALLINGCONV occaBuildKernelFromBinary(occaDevice device,
                                                                       const char *filename,
                                                                       const char *functionName);

  LIBOCCA_API occaKernel LIBOCCA_CALLINGCONV occaBuildKernelFromLoopy(occaDevice device,
                                                                      const char *filename,
                                                                      const char *functionName,
                                                                      occaKernelInfo info);

  LIBOCCA_API occaKernel LIBOCCA_CALLINGCONV occaBuildKernelFromFloopy(occaDevice device,
                                                                       const char *filename,
                                                                       const char *functionName,
                                                                       occaKernelInfo info);

  LIBOCCA_API occaMemory LIBOCCA_CALLINGCONV occaDeviceMalloc(occaDevice device,
                                                              uintptr_t bytes,
                                                              void *source);

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaDeviceFlush(occaDevice device);
  LIBOCCA_API void LIBOCCA_CALLINGCONV occaDeviceFinish(occaDevice device);

  LIBOCCA_API occaStream LIBOCCA_CALLINGCONV occaDeviceGenStream(occaDevice device);
  LIBOCCA_API occaStream LIBOCCA_CALLINGCONV occaDeviceGetStream(occaDevice device);
  LIBOCCA_API void       LIBOCCA_CALLINGCONV occaDeviceSetStream(occaDevice device, occaStream stream);

  LIBOCCA_API occaTag LIBOCCA_CALLINGCONV occaDeviceTagStream(occaDevice device);
  LIBOCCA_API double LIBOCCA_CALLINGCONV occaDeviceTimeBetweenTags(occaDevice device,
                                                                   occaTag startTag, occaTag endTag);

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaDeviceStreamFree(occaDevice device, occaStream stream);

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaDeviceFree(occaDevice device);
  //====================================


  //---[ Kernel ]-----------------------
  LIBOCCA_API const char* LIBOCCA_CALLINGCONV occaKernelMode(occaKernel kernel);

  LIBOCCA_API int LIBOCCA_CALLINGCONV occaKernelPreferredDimSize(occaKernel kernel);

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaKernelSetWorkingDims(occaKernel kernel,
                                                                int dims,
                                                                occaDim items,
                                                                occaDim groups);

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaKernelSetAllWorkingDims(occaKernel kernel,
                                                                   int dims,
                                                                   uintptr_t itemsX, uintptr_t itemsY, uintptr_t itemsZ,
                                                                   uintptr_t groupsX, uintptr_t groupsY, uintptr_t groupsZ);

  LIBOCCA_API double LIBOCCA_CALLINGCONV occaKernelTimeTaken(occaKernel kernel);

  LIBOCCA_API occaArgumentList LIBOCCA_CALLINGCONV occaGenArgumentList();

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaArgumentListClear(occaArgumentList list);

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaArgumentListFree(occaArgumentList list);

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaArgumentListAddArg(occaArgumentList list,
                                                              int argPos,
                                                              void *type);

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaKernelRun_(occaKernel kernel,
                                                      occaArgumentList list);

  OCCA_C_KERNEL_RUN_DECLARATIONS;

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaKernelFree(occaKernel kernel);

  LIBOCCA_API occaKernelInfo LIBOCCA_CALLINGCONV occaGenKernelInfo();

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaKernelInfoAddDefine(occaKernelInfo info,
                                                               const char *macro,
                                                               occaType value);

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaKernelInfoFree(occaKernelInfo info);
  //====================================


  //---[ Wrappers ]---------------------
#if OCCA_OPENCL_ENABLED
  LIBOCCA_API occaDevice LIBOCCA_CALLINGCONV occaWrapOpenCLDevice(cl_platform_id platformID,
                                                                  cl_device_id deviceID,
                                                                  cl_context context);
#endif

#if OCCA_CUDA_ENABLED
  LIBOCCA_API occaDevice LIBOCCA_CALLINGCONV occaWrapCudaDevice(CUdevice device, CUcontext context);
#endif

#if OCCA_COI_ENABLED
  LIBOCCA_API occaDevice LIBOCCA_CALLINGCONV occaWrapCoiDevice(COIENGINE coiDevice);
#endif

  LIBOCCA_API occaMemory LIBOCCA_CALLINGCONV occaDeviceWrapMemory(occaDevice device,
                                                                  void *handle_,
                                                                  const uintptr_t bytes);

  LIBOCCA_API occaStream LIBOCCA_CALLINGCONV occaDeviceWrapStream(occaDevice device, void *handle_);
  //====================================


  //---[ Memory ]-----------------------
  LIBOCCA_API const char* LIBOCCA_CALLINGCONV occaMemoryMode(occaMemory memory);

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaCopyMemToMem(occaMemory dest, occaMemory src,
                                                        const uintptr_t bytes,
                                                        const uintptr_t destOffset,
                                                        const uintptr_t srcOffset);

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaCopyPtrToMem(occaMemory dest, const void *src,
                                                        const uintptr_t bytes, const uintptr_t offset);

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaCopyMemToPtr(void *dest, occaMemory src,
                                                        const uintptr_t bytes, const uintptr_t offset);

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaAsyncCopyMemToMem(occaMemory dest, occaMemory src,
                                                             const uintptr_t bytes,
                                                             const uintptr_t destOffset,
                                                             const uintptr_t srcOffset);

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaAsyncCopyPtrToMem(occaMemory dest, const void *src,
                                                             const uintptr_t bytes, const uintptr_t offset);

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaAsyncCopyMemToPtr(void *dest, occaMemory src,
                                                             const uintptr_t bytes, const uintptr_t offset);

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaMemorySwap(occaMemory memoryA, occaMemory memoryB);

  LIBOCCA_API void LIBOCCA_CALLINGCONV occaMemoryFree(occaMemory memory);
  //====================================

#  ifdef __cplusplus
}
#  endif

#endif
