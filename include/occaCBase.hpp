#ifndef OCCA_CBASE_HEADER
#define OCCA_CBASE_HEADER

#include "stdlib.h"

#if OCCA_OPENCL_ENABLED
#  if   (OCCA_OS & LINUX_OS)
#    include <CL/cl.h>
#    include <CL/cl_gl.h>
#  elif (OCCA_OS & OSX_OS)
#    include <OpenCL/OpenCl.h>
#  else
#    include "CL/opencl.h"
#  endif
#endif

#if OCCA_CUDA_ENABLED
#  include <cuda.h>
#endif

#if OCCA_HSA_ENABLED
#  if   (OCCA_OS & LINUX_OS)
#  elif (OCCA_OS & OSX_OS)
#  else
#  endif
#endif

#include "occaDefines.hpp"

#if (OCCA_OS & (LINUX_OS | OSX_OS))
#  define OCCA_RFUNC
#  define OCCA_LFUNC
#else
#  define OCCA_RFUNC __stdcall
#  ifdef LIBOCCA_C_EXPORTS
//#define OCCA_LFUNC __declspec(dllexport)
#    define OCCA_LFUNC
#  else
//#define OCCA_LFUNC __declspec(dllimport)
#    define OCCA_LFUNC
#  endif
#endif

#define OCCA_TYPE_MEMORY 0
#define OCCA_TYPE_STRUCT 1
#define OCCA_TYPE_INT    2
#define OCCA_TYPE_UINT   3
#define OCCA_TYPE_CHAR   4
#define OCCA_TYPE_UCHAR  5
#define OCCA_TYPE_SHORT  6
#define OCCA_TYPE_USHORT 7
#define OCCA_TYPE_LONG   8
#define OCCA_TYPE_ULONG  9
#define OCCA_TYPE_FLOAT  10
#define OCCA_TYPE_DOUBLE 11
#define OCCA_TYPE_STRING 12
#define OCCA_TYPE_COUNT  13

#  ifdef __cplusplus
extern "C" {
#  endif

  typedef void* occaDevice;
  typedef void* occaKernel;

  typedef struct occaMemory_t* occaMemory;

  typedef struct occaType_t*         occaType;
  typedef struct occaArgumentList_t* occaArgumentList;

  typedef void* occaStream;

  typedef union occaStreamTag_t {
    double tagTime;
    void* otherStuff;
  } occaStreamTag;

  typedef void* occaDeviceInfo;
  typedef void* occaKernelInfo;

  typedef struct occaDim_t {
    uintptr_t x, y, z;
  } occaDim;

  //---[ Globals & Flags ]------------
  extern OCCA_LFUNC occaKernelInfo occaNoKernelInfo;

  extern OCCA_LFUNC const uintptr_t occaAutoSize;
  extern OCCA_LFUNC const uintptr_t occaNoOffset;

  extern OCCA_LFUNC const int occaUsingOKL;
  extern OCCA_LFUNC const int occaUsingOFL;
  extern OCCA_LFUNC const int occaUsingNative;

  OCCA_LFUNC void OCCA_RFUNC occaSetVerboseCompilation(const int value);
  //==================================


  //---[ TypeCasting ]------------------
  OCCA_LFUNC occaType OCCA_RFUNC occaInt(int value);
  OCCA_LFUNC occaType OCCA_RFUNC occaUInt(unsigned int value);

  OCCA_LFUNC occaType OCCA_RFUNC occaChar(char value);
  OCCA_LFUNC occaType OCCA_RFUNC occaUChar(unsigned char value);

  OCCA_LFUNC occaType OCCA_RFUNC occaShort(short value);
  OCCA_LFUNC occaType OCCA_RFUNC occaUShort(unsigned short value);

  OCCA_LFUNC occaType OCCA_RFUNC occaLong(long value);
  OCCA_LFUNC occaType OCCA_RFUNC occaULong(unsigned long value);

  OCCA_LFUNC occaType OCCA_RFUNC occaFloat(float value);
  OCCA_LFUNC occaType OCCA_RFUNC occaDouble(double value);

  OCCA_LFUNC occaType OCCA_RFUNC occaStruct(void *value, uintptr_t bytes);

  OCCA_LFUNC occaType OCCA_RFUNC occaString(char *str);
  //====================================


  //---[ Device ]-----------------------
  OCCA_LFUNC void OCCA_RFUNC occaPrintAvailableDevices();

  OCCA_LFUNC occaDeviceInfo OCCA_RFUNC occaCreateDeviceInfo();

  OCCA_LFUNC void OCCA_RFUNC occaDeviceInfoAppend(occaDeviceInfo info,
                                                  const char *key,
                                                  const char *value);

  OCCA_LFUNC void OCCA_RFUNC occaDeviceInfoAppendType(occaDeviceInfo info,
                                                      const char *key,
                                                      occaType value);

  OCCA_LFUNC void OCCA_RFUNC occaDeviceInfoFree(occaDeviceInfo info);

  OCCA_LFUNC occaDevice OCCA_RFUNC occaGetDevice(const char *infos);

  OCCA_LFUNC occaDevice OCCA_RFUNC occaGetDeviceFromInfo(occaDeviceInfo dInfo);

  OCCA_LFUNC occaDevice OCCA_RFUNC occaGetDeviceFromArgs(const char *mode,
                                                         int arg1, int arg2);

  OCCA_LFUNC const char* OCCA_RFUNC occaDeviceMode(occaDevice device);

  OCCA_LFUNC void OCCA_RFUNC occaDeviceSetCompiler(occaDevice device,
                                                   const char *compiler);

  OCCA_LFUNC void OCCA_RFUNC occaDeviceSetCompilerFlags(occaDevice device,
                                                        const char *compilerFlags);

  OCCA_LFUNC uintptr_t OCCA_RFUNC occaDeviceBytesAllocated(occaDevice device);

  OCCA_LFUNC occaKernel OCCA_RFUNC occaBuildKernel(occaDevice device,
                                                   const char *str,
                                                   const char *functionName,
                                                   occaKernelInfo info);

  OCCA_LFUNC occaKernel OCCA_RFUNC occaBuildKernelFromSource(occaDevice device,
                                                             const char *filename,
                                                             const char *functionName,
                                                             occaKernelInfo info);

  OCCA_LFUNC occaKernel OCCA_RFUNC occaBuildKernelFromString(occaDevice device,
                                                             const char *str,
                                                             const char *functionName,
                                                             occaKernelInfo info,
                                                             const int language);

  OCCA_LFUNC occaKernel OCCA_RFUNC occaBuildKernelFromBinary(occaDevice device,
                                                             const char *filename,
                                                             const char *functionName);

  OCCA_LFUNC occaKernel OCCA_RFUNC occaBuildKernelFromLoopy(occaDevice device,
                                                            const char *filename,
                                                            const char *functionName,
                                                            occaKernelInfo info);

  OCCA_LFUNC occaKernel OCCA_RFUNC occaBuildKernelFromFloopy(occaDevice device,
                                                             const char *filename,
                                                             const char *functionName,
                                                             occaKernelInfo info);

  OCCA_LFUNC occaMemory OCCA_RFUNC occaDeviceMalloc(occaDevice device,
                                                    uintptr_t bytes,
                                                    void *src);

  OCCA_LFUNC void* OCCA_RFUNC occaDeviceManagedAlloc(occaDevice device,
                                                     uintptr_t bytes,
                                                     void *src);

  OCCA_LFUNC void* OCCA_RFUNC occaDeviceUvaAlloc(occaDevice device,
                                                 uintptr_t bytes,
                                                 void *src);

  OCCA_LFUNC void* OCCA_RFUNC occaDeviceManagedUvaAlloc(occaDevice device,
                                                        uintptr_t bytes,
                                                        void *src);

  OCCA_LFUNC occaMemory OCCA_RFUNC occaDeviceMappedAlloc(occaDevice device,
                                                         uintptr_t bytes,
                                                         void *src);

  OCCA_LFUNC void* OCCA_RFUNC occaDeviceManagedMappedAlloc(occaDevice device,
                                                           uintptr_t bytes,
                                                           void *src);

  OCCA_LFUNC void OCCA_RFUNC occaDeviceFlush(occaDevice device);
  OCCA_LFUNC void OCCA_RFUNC occaDeviceFinish(occaDevice device);

  OCCA_LFUNC occaStream OCCA_RFUNC occaDeviceCreateStream(occaDevice device);
  OCCA_LFUNC occaStream OCCA_RFUNC occaDeviceGetStream(occaDevice device);
  OCCA_LFUNC void       OCCA_RFUNC occaDeviceSetStream(occaDevice device, occaStream stream);

  OCCA_LFUNC occaStreamTag OCCA_RFUNC occaDeviceTagStream(occaDevice device);
  OCCA_LFUNC double OCCA_RFUNC occaDeviceTimeBetweenTags(occaDevice device,
                                                         occaStreamTag startTag, occaStreamTag endTag);

  OCCA_LFUNC void OCCA_RFUNC occaDeviceStreamFree(occaDevice device, occaStream stream);

  OCCA_LFUNC void OCCA_RFUNC occaDeviceFree(occaDevice device);
  //====================================


  //---[ Kernel ]-----------------------
  OCCA_LFUNC const char* OCCA_RFUNC occaKernelMode(occaKernel kernel);
  OCCA_LFUNC const char* OCCA_RFUNC occaKernelName(occaKernel kernel);

  OCCA_LFUNC occaDevice OCCA_RFUNC occaKernelGetDevice(occaKernel kernel);

  OCCA_LFUNC uintptr_t OCCA_RFUNC occaKernelMaximumInnerDimSize(occaKernel kernel);
  OCCA_LFUNC int       OCCA_RFUNC occaKernelPreferredDimSize(occaKernel kernel);

  OCCA_LFUNC void OCCA_RFUNC occaKernelSetWorkingDims(occaKernel kernel,
                                                      int dims,
                                                      occaDim items,
                                                      occaDim groups);

  OCCA_LFUNC void OCCA_RFUNC occaKernelSetAllWorkingDims(occaKernel kernel,
                                                         int dims,
                                                         uintptr_t itemsX, uintptr_t itemsY, uintptr_t itemsZ,
                                                         uintptr_t groupsX, uintptr_t groupsY, uintptr_t groupsZ);

  OCCA_LFUNC occaArgumentList OCCA_RFUNC occaCreateArgumentList();

  OCCA_LFUNC void OCCA_RFUNC occaArgumentListClear(occaArgumentList list);

  OCCA_LFUNC void OCCA_RFUNC occaArgumentListFree(occaArgumentList list);

  OCCA_LFUNC void OCCA_RFUNC occaArgumentListAddArg(occaArgumentList list,
                                                    int argPos,
                                                    void *type);

#define OCCA_ARG_COUNT(...) OCCA_ARG_COUNT2(__VA_ARGS__,                \
                                            50,49,48,47,46,45,44,43,42,41, \
                                            40,39,38,37,36,35,34,33,32,31, \
                                            30,29,28,27,26,25,24,23,22,21, \
                                            20,19,18,17,16,15,14,13,12,11, \
                                            10,9,8,7,6,5,4,3,2,1)

#define OCCA_ARG_COUNT2(KERNEL,                                         \
                        _1,_2,_3,_4,_5,_6,_7,_8,_9,_10,                 \
                        _11,_12,_13,_14,_15,_16,_17,_18,_19,_20,        \
                        _21,_22,_23,_24,_25,_26,_27,_28,_29,_30,        \
                        _31,_32,_33,_34,_35,_36,_37,_38,_39,_40,        \
                        _41,_42,_43,_44,_45,_46,_47,_48,_49,_50, N, ...) N

#define OCCA_C_RUN_KERNEL1(...) OCCA_C_RUN_KERNEL2(__VA_ARGS__)
#define OCCA_C_RUN_KERNEL2(...) OCCA_C_RUN_KERNEL3(__VA_ARGS__)
#define OCCA_C_RUN_KERNEL3(N, kernel, ...) occaKernelRun##N(kernel, __VA_ARGS__)

#define occaKernelRun(...) OCCA_C_RUN_KERNEL1(OCCA_ARG_COUNT(__VA_ARGS__), __VA_ARGS__)

  OCCA_LFUNC void OCCA_RFUNC occaKernelRun_(occaKernel kernel,
                                            occaArgumentList list);

#include "operators/occaCKernelOperators.hpp"

  OCCA_LFUNC void OCCA_RFUNC occaKernelFree(occaKernel kernel);

  OCCA_LFUNC occaKernelInfo OCCA_RFUNC occaCreateKernelInfo();

  OCCA_LFUNC void OCCA_RFUNC occaKernelInfoAddDefine(occaKernelInfo info,
                                                     const char *macro,
                                                     occaType value);

  OCCA_LFUNC void OCCA_RFUNC occaKernelInfoAddInclude(occaKernelInfo info,
                                                      const char *filename);

  OCCA_LFUNC void OCCA_RFUNC occaKernelInfoFree(occaKernelInfo info);
  //====================================


  //---[ Wrappers ]---------------------
#if OCCA_OPENCL_ENABLED
  OCCA_LFUNC occaDevice OCCA_RFUNC occaWrapOpenCLDevice(cl_platform_id platformID,
                                                        cl_device_id deviceID,
                                                        cl_context context);
#endif

#if OCCA_CUDA_ENABLED
  OCCA_LFUNC occaDevice OCCA_RFUNC occaWrapCudaDevice(CUdevice device, CUcontext context);
#endif

#if OCCA_HSA_ENABLED
  OCCA_LFUNC occaDevice OCCA_RFUNC occaWrapHSADevice();
#endif

#if OCCA_COI_ENABLED
  OCCA_LFUNC occaDevice OCCA_RFUNC occaWrapCoiDevice(COIENGINE coiDevice);
#endif

  OCCA_LFUNC occaMemory OCCA_RFUNC occaDeviceWrapMemory(occaDevice device,
                                                        void *handle_,
                                                        const uintptr_t bytes);

  OCCA_LFUNC occaStream OCCA_RFUNC occaDeviceWrapStream(occaDevice device, void *handle_);
  //====================================


  //---[ Memory ]-----------------------
  OCCA_LFUNC const char* OCCA_RFUNC occaMemoryMode(occaMemory memory);

  OCCA_LFUNC void* OCCA_RFUNC occaMemoryGetMemoryHandle(occaMemory mem);
  OCCA_LFUNC void* OCCA_RFUNC occaMemoryGetMappedPointer(occaMemory mem);
  OCCA_LFUNC void* OCCA_RFUNC occaMemoryGetTextureHandle(occaMemory mem);

  OCCA_LFUNC void OCCA_RFUNC occaMemcpy(void *dest, void *src,
                                        const uintptr_t bytes);

  OCCA_LFUNC void OCCA_RFUNC occaCopyMemToMem(occaMemory dest, occaMemory src,
                                              const uintptr_t bytes,
                                              const uintptr_t destOffset,
                                              const uintptr_t srcOffset);

  OCCA_LFUNC void OCCA_RFUNC occaCopyPtrToMem(occaMemory dest, const void *src,
                                              const uintptr_t bytes, const uintptr_t offset);

  OCCA_LFUNC void OCCA_RFUNC occaCopyMemToPtr(void *dest, occaMemory src,
                                              const uintptr_t bytes, const uintptr_t offset);

  OCCA_LFUNC void OCCA_RFUNC occaAsyncCopyMemToMem(occaMemory dest, occaMemory src,
                                                   const uintptr_t bytes,
                                                   const uintptr_t destOffset,
                                                   const uintptr_t srcOffset);

  OCCA_LFUNC void OCCA_RFUNC occaAsyncCopyPtrToMem(occaMemory dest, const void *src,
                                                   const uintptr_t bytes, const uintptr_t offset);

  OCCA_LFUNC void OCCA_RFUNC occaAsyncCopyMemToPtr(void *dest, occaMemory src,
                                                   const uintptr_t bytes, const uintptr_t offset);

  OCCA_LFUNC void OCCA_RFUNC occaMemoryFree(occaMemory memory);
  //====================================

#  ifdef __cplusplus
}
#  endif

#endif
