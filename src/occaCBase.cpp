#include "occaBase.hpp"
#include "occaCBase.hpp"

#  ifdef __cplusplus
extern "C" {
#  endif
  //---[ General ]----------------------
  void occaSetOmpCompiler(const char *compiler){
    occa::ompCompiler = compiler;
  }

  void occaSetOmpCompilerFlags(const char *compilerFlags){
    occa::ompCompilerFlags = compilerFlags;
  }


  void occaSetCudaCompiler(const char *compiler){
    occa::cudaCompiler = compiler;
  }

  void occaSetCudaCompilerFlags(const char *compilerFlags){
    occa::cudaCompilerFlags = compilerFlags;
  }

  //====================================


  //---[ Device ]-----------------------
  void occaDeviceSetOmpCompiler(occaDevice device,
                                const char *compiler){
    occa::device &device_ = *((occa::device*) device);
    device_.ompCompiler = compiler;
  }


  void occaDeviceSetOmpCompilerFlags(occaDevice device,
                                     const char *compilerFlags){
    occa::device &device_ = *((occa::device*) device);
  }


  void occaDeviceSetCudaCompiler(occaDevice device,
                                 const char *compiler){
    occa::device &device_ = *((occa::device*) device);
  }


  void occaDeviceSetCudaCompilerFlags(occaDevice device,
                                      const char *compilerFlags){
    occa::device &device_ = *((occa::device*) device);
  }


  const char* occaDeviceMode(occaDevice device){
    occa::device &device_ = *((occa::device*) device);

    return device_.mode.c_str();
  }


  occaDevice occaGetDevice(const char *mode,
                           int platformID, int deviceID){
    occa::device *device = new occa::device();

    device->setup(mode, platformID, deviceID);

    return (occaDevice) device;
  }


  occaKernel occaBuildKernelFromSource(occaDevice device,
                                       const char *filename,
                                       const char *functionName,
                                       occaKernelInfo info){
    occa::device &device_   = *((occa::device*) device);
    occa::kernelInfo &info_ = *((occa::kernelInfo*) info);

    occa::kernel *kernel = new occa::kernel();

    *kernel = device_.buildKernelFromBinary(filename,
                                            functionName,
                                            info_);

    return (occaKernel) kernel;
  }


  occaKernel occaBuildKernelFromBinary(occaDevice device,
                                       const char *filename,
                                       const char *functionName){
    occa::device &device_ = *((occa::device*) device);

    occa::kernel *kernel = new occa::kernel();

    *kernel = device_.buildKernelFromBinary(filename, functionName);

    return (occaKernel) kernel;
  }


  occaMemory occaMalloc(occaDevice device,
                        size_t bytes,
                        const void *source){
    occa::device &device_ = *((occa::device*) device);

    occa::memory *memory = new occa::memory();

    *memory = device_.malloc(bytes, source);

    return (occaMemory) memory;
  }


  occaStream occaGenStream(occaDevice device){
    occa::device &device_ = *((occa::device*) device);

    return (occaStream) device_.genStream();
  }

  occaStream occaGetStream(occaDevice device){
    occa::device &device_ = *((occa::device*) device);

    return (occaStream) device_.getStream();
  }

  void occaSetStream(occaDevice device, occaStream stream){
    occa::device &device_ = *((occa::device*) device);
    occa::stream &stream_ = *((occa::stream*) stream);

    device_.setStream(stream_);
  }


  void occaDeviceFree(occaDevice device){
    occa::device &device_ = *((occa::device*) device);

    device_.free();

    delete device;
  }

  //====================================


  //---[ Kernel ]-----------------------
  const char* occaKernelMode(occaKernel kernel){
    occa::kernel &kernel_ = *((occa::kernel*) kernel);

    return kernel_.mode.c_str();
  }

  int occaKernelPerferredDimSize(occaKernel kernel){
    occa::kernel &kernel_ = *((occa::kernel*) kernel);

    return kernel_.preferredDimSize();
  }

  void occaKernelSetWorkingDims(occaKernel kernel,
                                int dims,
                                occaDim itemsPerGroup,
                                occaDim groups){
    occa::kernel &kernel_ = *((occa::kernel*) kernel);

    kernel_.setWorkingDims(dims, itemsPerGroup, groups);
  }


  double occaKernelTimeTaken(occaKernel kernel){
    occa::kernel &kernel_ = *((occa::kernel*) kernel);

    return kernel_.timeTaken();
  }


  void occaKernelFree(occaKernel kernel){
    occa::kernel &kernel_ = *((occa::kernel*) kernel);

    kernel_.free();

    delete kernel;
  }

  // Operators

  //====================================


  //---[ Memory ]-----------------------
  const char* occaMemoryMode(occaMemory memory){
    occa::memory &memory_ = *((occa::memory*) memory);

    return memory_.mode.c_str();
  }

  // Copies

  void occaMemorySwap(occaMemory memoryA, occaMemory memoryB){
    occa::memory &memoryA_ = *((occa::memory*) memoryA);
    occa::memory &memoryA_ = *((occa::memory*) memoryB);

    memoryA_.swap(memoryB_);
  }


  void occaMemoryFree(occaMemory memory){
    occa::memory &memory_ = *((occa::memory*) memory);

    memory_.free();

    delete memory;
  }
  //====================================

#  ifdef __cplusplus
}
#  endif
