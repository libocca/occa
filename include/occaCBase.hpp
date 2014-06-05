#ifndef OCCA_CBASE_HEADER
#define OCCA_CBASE_HEADER

#  ifdef __cplusplus
extern "C" {
#  endif

  typedef void* occaDevice;
  typedef void* occaKernel;
  typedef void* occaMemory;

  typedef void* occaStream;

  typedef void* occaKernelInfo;

  typedef struct occaDim_t {
    size_t x, y, z;
  } occaDim;

  //---[ General ]----------------------
  void occaSetOmpCompiler(const char *compiler);
  void occaSetOmpCompilerFlags(const char *compilerFlags);

  void occaSetCudaCompiler(const char *compiler);
  void occaSetCudaCompilerFlags(const char *compilerFlags);
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

  void occaKernelFree(occaKernel kernel);

  // Operators

  //====================================


  //---[ Memory ]-----------------------
  const char* occaMemoryMode(occaMemory memory);

  // Copies

  void occaMemorySwap(occaMemory memoryA, occaMemory memoryB);

  void occaMemoryFree(occaMemory memory);
  //====================================

#  ifdef __cplusplus
}
#  endif

#endif
