#ifndef OCCA_CBASE_HEADER
#define OCCA_CBASE_HEADER

#include "ocl_preprocessor.hpp"

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
#define OCCA_TYPE_COUNT  11

#define OCCA_ARG_COUNT(...) OCCA_ARG_COUNT2(__VA_ARGS__, 25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1)
#define OCCA_ARG_COUNT2(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,_17,_18,_19,_20,_21,_22,_23,_24,_25, N, ...) N

#define occaKernelRun(...) OCCA_C_RUN_KERNEL1( OCL_SUB(OCCA_ARG_COUNT(__VA_ARGS__), 1) , __VA_ARGS__)
#define OCCA_C_RUN_KERNEL1(...) OCCA_C_RUN_KERNEL2(__VA_ARGS__)
#define OCCA_C_RUN_KERNEL2(...) OCCA_C_RUN_KERNEL3(__VA_ARGS__)
#define OCCA_C_RUN_KERNEL3(N, kernel, ...) occaKernelRun##N(kernel, __VA_ARGS__)

//---[ Declarations ]-------------------

#define OCCA_C_KERNEL_RUN_DECLARATION_ARGS(N) , void *arg##N
#define OCCA_C_KERNEL_RUN_DECLARATION(N)                                \
  void occaKernelRun##N(occaKernel kernel OCL_FOR(1, N, OCCA_C_KERNEL_RUN_DECLARATION_ARGS));

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

  typedef void* occaKernelInfo;

  typedef struct occaDim_t {
    size_t x, y, z;
  } occaDim;

  extern occaKernelInfo occaNoKernelInfo;

  extern size_t occaAutoSize;
  extern size_t occaNoOffset;

  extern const size_t occaTypeSize[OCCA_TYPE_COUNT];

  //---[ General ]----------------------
  void occaSetOmpCompiler(const char *compiler);
  void occaSetOmpCompilerFlags(const char *compilerFlags);

  void occaSetCudaCompiler(const char *compiler);
  void occaSetCudaCompilerFlags(const char *compilerFlags);
  //====================================


  //---[ TypeCasting ]------------------
  occaType occaInt(int value);
  occaType occaUInt(unsigned int value);

  occaType occaChar(char value);
  occaType occaUChar(unsigned char value);

  occaType occaShort(short value);
  occaType occaUShort(unsigned short value);

  occaType occaLong(long value);
  occaType occaULong(unsigned long value);

  occaType occaFloat(float value);
  occaType occaDouble(double value);
  //====================================


  //---[ Device ]-----------------------
  const char* occaDeviceMode(occaDevice device);

  void occaDeviceSetOmpCompiler(occaDevice device,
                                const char *compiler);

  void occaDeviceSetOmpCompilerFlags(occaDevice device,
                                     const char *compilerFlags);

  void occaDeviceSetCudaCompiler(occaDevice device,
                                 const char *compiler);

  void occaDeviceSetCudaCompilerFlags(occaDevice device,
                                      const char *compilerFlags);

  occaDevice occaGetDevice(const char *mode,
                           int platformID, int deviceID);

  occaKernel occaBuildKernelFromSource(occaDevice device,
                                       const char *filename,
                                       const char *functionName,
                                       occaKernelInfo info);

  occaKernel occaBuildKernelFromBinary(occaDevice device,
                                       const char *filename,
                                       const char *functionName);

  occaMemory occaDeviceMalloc(occaDevice device,
                              size_t bytes,
                              void *source);

  occaStream occaGenStream(occaDevice device);
  occaStream occaGetStream(occaDevice device);
  void       occaSetStream(occaDevice device, occaStream stream);

  void occaDeviceFree(occaDevice device);
  //====================================


  //---[ Kernel ]-----------------------
  const char* occaKernelMode(occaKernel kernel);

  int occaKernelPreferredDimSize(occaKernel kernel);

  void occaKernelSetWorkingDims(occaKernel kernel,
                                int dims,
                                occaDim items,
                                occaDim groups);

  double occaKernelTimeTaken(occaKernel kernel);

  occaArgumentList occaGenArgumentList();

  void occaArgumentListClear(occaArgumentList list);

  void occaArgumentListFree(occaArgumentList list);

  void occaArgumentListAddArg(occaArgumentList list,
                              int argPos,
                              void *type);

  void occaKernelRun_(occaKernel kernel,
                      occaArgumentList list);

  OCCA_C_KERNEL_RUN_DECLARATIONS;

  void occaKernelFree(occaKernel kernel);

  occaKernelInfo occaGenKernelInfo();

  void occaKernelInfoAddDefine(occaKernelInfo info,
                               const char *macro,
                               occaType value);

  void occaKernelInfoFree(occaKernelInfo info);
  //====================================


  //---[ Memory ]-----------------------
  const char* occaMemoryMode(occaMemory memory);

  void occaCopyMemToMem(occaMemory dest, occaMemory src,
                        const size_t bytes, const size_t offset);

  void occaCopyPtrToMem(occaMemory dest, void *src,
                        const size_t bytes, const size_t offset);

  void occaCopyMemToPtr(void *dest, occaMemory src,
                        const size_t bytes, const size_t offset);

  void occaAsyncCopyMemToMem(occaMemory dest, occaMemory src,
                             const size_t bytes, const size_t offset);

  void occaAsyncCopyPtrToMem(occaMemory dest, void *src,
                             const size_t bytes, const size_t offset);

  void occaAsyncCopyMemToPtr(void *dest, occaMemory src,
                             const size_t bytes, const size_t offset);

  void occaMemorySwap(occaMemory memoryA, occaMemory memoryB);

  void occaMemoryFree(occaMemory memory);
  //====================================

#  ifdef __cplusplus
}
#  endif

#endif
