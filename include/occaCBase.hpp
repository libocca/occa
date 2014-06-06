#ifndef OCCA_CBASE_HEADER
#define OCCA_CBASE_HEADER

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
  void occaDeviceSetOmpCompiler(occaDevice device,
                                const char *compiler);

  void occaDeviceSetOmpCompilerFlags(occaDevice device,
                                     const char *compilerFlags);

  void occaDeviceSetCudaCompiler(occaDevice device,
                                 const char *compiler);

  void occaDeviceSetCudaCompilerFlags(occaDevice device,
                                      const char *compilerFlags);

  const char* occaDeviceMode(occaDevice device);

  occaDevice occaGetDevice(const char *mode,
                           int platformID, int deviceID);

  occaKernel occaBuildKernelFromSource(occaDevice device,
                                       const char *filename,
                                       const char *functionName,
                                       occaKernelInfo info);

  occaKernel occaBuildKernelFromBinary(occaDevice device,
                                       const char *filename,
                                       const char *functionName);

  occaMemory occaMalloc(occaDevice device,
                        size_t bytes,
                        void *source);

  occaStream occaGenStream(occaDevice device);
  occaStream occaGetStream(occaDevice device);
  void       occaSetStream(occaDevice device, occaStream stream);

  void occaDeviceFree(occaDevice device);
  //====================================


  //---[ Kernel ]-----------------------
  const char* occaKernelMode(occaKernel kernel);

  int occaKernelPerferredDimSize(occaKernel kernel);

  void occaKernelSetWorkingDims(occaKernel kernel,
                                int dims,
                                occaDim items,
                                occaDim groups);

  double occaKernelTimeTaken(occaKernel kernel);

  void occaAddArgument(occaArgumentList list,
                       occaMemory type);

  void occaRunKernel_(occaKernel kernel,
                      occaArgumentList list);

  void occaKernelFree(occaKernel kernel);

  occaKernelInfo occaGenKernelInfo();

  void occaKernelInfoAddDefine(occaKernelInfo info,
                               const char *macro,
                               occaType value);

  void occaKernelInfoFree(occaKernelInfo info);
  //====================================


  //---[ Memory ]-----------------------
  const char* occaMemoryMode(occaMemory memory);

  void occaCopyFromMem(occaMemory dest, occaMemory src,
                       const size_t bytes, const size_t offset);

  void occaCopyFromPtr(occaMemory dest, void *src,
                       const size_t bytes, const size_t offset);

  void occaCopyToMem(occaMemory dest, occaMemory src,
                     const size_t bytes, const size_t offset);

  void occaCopyToPtr(void *dest, occaMemory src,
                     const size_t bytes, const size_t offset);

  void occaAsyncCopyFromMem(occaMemory dest, occaMemory src,
                            const size_t bytes, const size_t offset);

  void occaAsyncCopyFromPtr(occaMemory dest, void * src,
                            const size_t bytes, const size_t offset);

  void occaAsyncCopyToMem(occaMemory dest, occaMemory src,
                          const size_t bytes, const size_t offset);

  void occaAsyncCopyToPtr(void *dest, occaMemory src,
                          const size_t bytes, const size_t offset);

  void occaMemorySwap(occaMemory memoryA, occaMemory memoryB);

  void occaMemoryFree(occaMemory memory);
  //====================================

#  ifdef __cplusplus
}
#  endif

#endif
