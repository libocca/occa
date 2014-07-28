#ifndef OCCA_CBASE_HEADER
#define OCCA_CBASE_HEADER

#include "ocl_preprocessor.hpp"

#include "stdlib.h"

#ifdef WIN32
#ifdef LIBOCCA_C_EXPORTS
#define LIBOCCA_API __declspec(dllexport)
#else
#define LIBOCCA_API __declspec(dllimport)
#endif
#else
#define LIBOCCA_API  
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
  LIBOCCA_API void occaKernelRun##N(occaKernel kernel OCL_FOR(1, N, OCCA_C_KERNEL_RUN_DECLARATION_ARGS));

#define OCCA_C_KERNEL_RUN_DECLARATIONS          \
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

#define OCCA_C_KERNEL_RUN_DEFINITION(N)   \
  void occaKernelRun##N(occaKernel kernel OCL_FOR(1, N, OCCA_C_KERNEL_RUN_DECLARATION_ARGS)){ \
    occa::kernel &__occa_kernel__  = *((occa::kernel*) kernel);         \
    __occa_kernel__.clearArgumentList();                                \
                                                                        \
    OCL_FOR(1, N, OCCA_C_KERNEL_RUN_ADD_ARG);                           \
                                                                        \
    __occa_kernel__.runFromArguments();                                 \
  }

#define OCCA_C_KERNEL_RUN_DEFINITIONS          \
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

  union occaTag {
    double tagTime;
    void* otherStuff;
  };

  typedef void* occaKernelInfo;

  typedef struct occaDim_t {
    uintptr_t x, y, z;
  } occaDim;

  extern LIBOCCA_API occaKernelInfo occaNoKernelInfo;

  extern LIBOCCA_API const uintptr_t occaAutoSize;
  extern LIBOCCA_API const uintptr_t occaNoOffset;

  extern LIBOCCA_API const uintptr_t occaTypeSize[OCCA_TYPE_COUNT];


  //---[ TypeCasting ]------------------
  LIBOCCA_API occaType occaInt(int value);
  LIBOCCA_API occaType occaUInt(unsigned int value);

  LIBOCCA_API occaType occaChar(char value);
  LIBOCCA_API occaType occaUChar(unsigned char value);

  LIBOCCA_API occaType occaShort(short value);
  LIBOCCA_API occaType occaUShort(unsigned short value);

  LIBOCCA_API occaType occaLong(long value);
  LIBOCCA_API occaType occaULong(unsigned long value);

  LIBOCCA_API occaType occaFloat(float value);
  LIBOCCA_API occaType occaDouble(double value);

  LIBOCCA_API occaType occaString(char *str);
  //====================================


  //---[ Device ]-----------------------
  LIBOCCA_API const char* occaDeviceMode(occaDevice device);

  LIBOCCA_API void occaDeviceSetCompiler(occaDevice device,
                                         const char *compiler);

  LIBOCCA_API void occaDeviceSetCompilerFlags(occaDevice device,
                                              const char *compilerFlags);

  LIBOCCA_API occaDevice occaGetDevice(const char *mode,
                                       int arg1, int arg2);

  LIBOCCA_API occaKernel occaBuildKernelFromSource(occaDevice device,
                                                   const char *filename,
                                                   const char *functionName,
                                                   occaKernelInfo info);

  LIBOCCA_API occaKernel occaBuildKernelFromBinary(occaDevice device,
                                                   const char *filename,
                                                   const char *functionName);

  LIBOCCA_API occaKernel occaBuildKernelFromLoopy(occaDevice device,
                                                  const char *filename,
                                                  const char *functionName,
                                                  const char *pythonCode);

  LIBOCCA_API occaMemory occaDeviceMalloc(occaDevice device,
                                          uintptr_t bytes,
                                          void *source);

  LIBOCCA_API void occaDeviceFlush(occaDevice device);
  LIBOCCA_API void occaDeviceFinish(occaDevice device);

  LIBOCCA_API occaStream occaDeviceGenStream(occaDevice device);
  LIBOCCA_API occaStream occaDeviceGetStream(occaDevice device);
  LIBOCCA_API void       occaDeviceSetStream(occaDevice device, occaStream stream);

  LIBOCCA_API occaTag occaDeviceTagStream(occaDevice device);
  LIBOCCA_API double occaDeviceTimeBetweenTags(occaDevice device,
                                               occaTag startTag, occaTag endTag);

  LIBOCCA_API void occaDeviceStreamFree(occaDevice device, occaStream stream);

  LIBOCCA_API void occaDeviceFree(occaDevice device);
  //====================================


  //---[ Kernel ]-----------------------
  LIBOCCA_API const char* occaKernelMode(occaKernel kernel);

  LIBOCCA_API int occaKernelPreferredDimSize(occaKernel kernel);

  LIBOCCA_API void occaKernelSetWorkingDims(occaKernel kernel,
                                            int dims,
                                            occaDim items,
                                            occaDim groups);

  LIBOCCA_API void occaKernelSetAllWorkingDims(occaKernel kernel,
                                               int dims,
                                               uintptr_t itemsX, uintptr_t itemsY, uintptr_t itemsZ,
                                               uintptr_t groupsX, uintptr_t groupsY, uintptr_t groupsZ);

  LIBOCCA_API double occaKernelTimeTaken(occaKernel kernel);

  LIBOCCA_API occaArgumentList occaGenArgumentList();

  LIBOCCA_API void occaArgumentListClear(occaArgumentList list);

  LIBOCCA_API void occaArgumentListFree(occaArgumentList list);

  LIBOCCA_API void occaArgumentListAddArg(occaArgumentList list,
                                          int argPos,
                                          void *type);

  LIBOCCA_API void occaKernelRun_(occaKernel kernel,
                                       occaArgumentList list);

  OCCA_C_KERNEL_RUN_DECLARATIONS;

  LIBOCCA_API void occaKernelFree(occaKernel kernel);

  LIBOCCA_API occaKernelInfo occaGenKernelInfo();

  LIBOCCA_API void occaKernelInfoAddDefine(occaKernelInfo info,
                                           const char *macro,
                                           occaType value);

  LIBOCCA_API void occaKernelInfoFree(occaKernelInfo info);
  //====================================


  //---[ Memory ]-----------------------
  LIBOCCA_API const char* occaMemoryMode(occaMemory memory);

  LIBOCCA_API void occaCopyMemToMem(occaMemory dest, occaMemory src,
                                    const uintptr_t bytes,
                                    const uintptr_t destOffset,
                                    const uintptr_t srcOffset);

  LIBOCCA_API void occaCopyPtrToMem(occaMemory dest, const void *src,
                                    const uintptr_t bytes, const uintptr_t offset);

  LIBOCCA_API void occaCopyMemToPtr(void *dest, occaMemory src,
                                    const uintptr_t bytes, const uintptr_t offset);

  LIBOCCA_API void occaAsyncCopyMemToMem(occaMemory dest, occaMemory src,
                                         const uintptr_t bytes,
                                         const uintptr_t destOffset,
                                         const uintptr_t srcOffset);

  LIBOCCA_API void occaAsyncCopyPtrToMem(occaMemory dest, const void *src,
                                         const uintptr_t bytes, const uintptr_t offset);

  LIBOCCA_API void occaAsyncCopyMemToPtr(void *dest, occaMemory src,
                                         const uintptr_t bytes, const uintptr_t offset);

  LIBOCCA_API void occaMemorySwap(occaMemory memoryA, occaMemory memoryB);

  LIBOCCA_API void occaMemoryFree(occaMemory memory);
  //====================================

#  ifdef __cplusplus
}
#  endif

#endif
