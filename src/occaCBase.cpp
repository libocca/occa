#include "occaBase.hpp"
#include "occaCBase.hpp"

#  ifdef __cplusplus
extern "C" {
#  endif

  struct occaMemory_t {
    int type;
    void *ptr;
  };

  struct occaType_t {
    int type;
    occa::kernelArg_t value;
  };

  struct occaArgumentList_t {
    int argc;
    occaMemory argv[100];
  };

  occaKernelInfo occaNoKernelInfo = NULL;

  size_t occaAutoSize = 0;
  size_t occaNoOffset = 0;

  const size_t occaTypeSize[OCCA_TYPE_COUNT] = {
    sizeof(void*),
    sizeof(int),
    sizeof(unsigned int),
    sizeof(char),
    sizeof(unsigned char),
    sizeof(short),
    sizeof(unsigned short),
    sizeof(long),
    sizeof(unsigned long),
    sizeof(float),
    sizeof(double)
  };

  //---[ TypeCasting ]------------------
  occaType occaInt(int value){
    occaType_t *type = new occaType_t;

    type->type       = OCCA_TYPE_INT;
    type->value.int_ = value;

    return (occaType) type;
  }

  occaType occaUInt(unsigned int value){
    occaType_t *type = new occaType_t;

    type->type        = OCCA_TYPE_UINT;
    type->value.uint_ = value;

    return (occaType) type;
  }

  occaType occaChar(char value){
    occaType_t *type = new occaType_t;

    type->type        = OCCA_TYPE_CHAR;
    type->value.char_ = value;

    return (occaType) type;
  }

  occaType occaUChar(unsigned char value){
    occaType_t *type = new occaType_t;

    type->type         = OCCA_TYPE_UCHAR;
    type->value.uchar_ = value;

    return (occaType) type;
  }

  occaType occaShort(short value){
    occaType_t *type = new occaType_t;

    type->type         = OCCA_TYPE_SHORT;
    type->value.short_ = value;

    return (occaType) type;
  }

  occaType occaUShort(unsigned short value){
    occaType_t *type = new occaType_t;

    type->type          = OCCA_TYPE_USHORT;
    type->value.ushort_ = value;

    return (occaType) type;
  }

  occaType occaLong(long value){
    occaType_t *type = new occaType_t;

    type->type        = OCCA_TYPE_LONG;
    type->value.long_ = value;

    return (occaType) type;
  }

  occaType occaULong(unsigned long value){
    occaType_t *type = new occaType_t;

    type->type          = OCCA_TYPE_ULONG;
    type->value.size_t_ = value;

    return (occaType) type;
  }

  occaType occaFloat(float value){
    occaType_t *type = new occaType_t;

    type->type         = OCCA_TYPE_FLOAT;
    type->value.float_ = value;

    return (occaType) type;
  }

  occaType occaDouble(double value){
    occaType_t *type = new occaType_t;

    type->type          = OCCA_TYPE_DOUBLE;
    type->value.double_ = value;

    return (occaType) type;
  }
  //====================================

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

    return device_.mode().c_str();
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

    occa::kernel *kernel = new occa::kernel();

    if(info != occaNoKernelInfo){
      occa::kernelInfo &info_ = *((occa::kernelInfo*) info);

      *kernel = device_.buildKernelFromSource(filename,
                                              functionName,
                                              info_);
    }
    else{
      *kernel = device_.buildKernelFromSource(filename,
                                              functionName);
    }

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


  occaMemory occaDeviceMalloc(occaDevice device,
                        size_t bytes,
                        void *source){
    occa::device &device_ = *((occa::device*) device);

    occaMemory_t *memory = new occaMemory_t();

    memory->type = OCCA_TYPE_MEMORY;
    memory->ptr  = new occa::memory();

    *((occa::memory*) memory->ptr) = device_.malloc(bytes, source);

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

    delete (occa::device*) device;
  }

  //====================================


  //---[ Kernel ]-----------------------
  const char* occaKernelMode(occaKernel kernel){
    occa::kernel &kernel_ = *((occa::kernel*) kernel);

    return kernel_.mode().c_str();
  }

  int occaKernelPerferredDimSize(occaKernel kernel){
    occa::kernel &kernel_ = *((occa::kernel*) kernel);

    return kernel_.preferredDimSize();
  }

  void occaKernelSetWorkingDims(occaKernel kernel,
                                int dims,
                                occaDim items,
                                occaDim groups){
    occa::kernel &kernel_ = *((occa::kernel*) kernel);

    kernel_.setWorkingDims(dims,
                           occa::dim(items.x, items.y, items.z),
                           occa::dim(groups.x, groups.y, groups.z));
  }


  double occaKernelTimeTaken(occaKernel kernel){
    occa::kernel &kernel_ = *((occa::kernel*) kernel);

    return kernel_.timeTaken();
  }

  occaArgumentList occaGenArgumentList(){
    occaArgumentList_t *list = new occaArgumentList_t();
    list->argc = 0;

    return (occaArgumentList) list;
  }

  void occaArgumentListClear(occaArgumentList list){
    occaArgumentList_t &list_ = *((occaArgumentList_t*) list);

    list_.argc = 0;
  }

  void occaArgumentListFree(occaArgumentList list){
    delete (occaArgumentList_t*) list;
  }



  void occaAddArgument(occaArgumentList list,
                       occaMemory type){
    occaArgumentList_t &list_ = *((occaArgumentList_t*) list);

    list_.argv[list_.argc++] = type;
  }

  // Note the _
  //   Macro that is called > API function that is never seen
  void occaRunKernel_(occaKernel kernel,
                      occaArgumentList list){
    occa::kernel &kernel_     = *((occa::kernel*) kernel);
    occaArgumentList_t &list_ = *((occaArgumentList_t*) list);

    for(int i = 0; i < list_.argc; ++i){
      occaMemory_t &memory_ = *((occaMemory_t*) list_.argv[i]);

      if(memory_.type == OCCA_TYPE_MEMORY){
        kernel_.addArgument(i, occa::kernelArg(memory_.ptr));
      }
      else{
        occaType_t &type_ = *((occaType_t*) list_.argv[i]);

        kernel_.addArgument(i, occa::kernelArg(type_.value,
                                               occaTypeSize[memory_.type],
                                               false));
      }
    }

    kernel_.runFromArguments();
  }

  void occaKernelFree(occaKernel kernel){
    occa::kernel &kernel_ = *((occa::kernel*) kernel);

    kernel_.free();

    delete (occa::kernel*) kernel;
  }

  occaKernelInfo occaGenKernelInfo(){
    occa::kernelInfo *info = new occa::kernelInfo();

    return (occaKernelInfo) info;

  }

  void occaKernelInfoAddDefine(occaKernelInfo info,
                               const char *macro,
                               occaType value){
    occa::kernelInfo &info_   = *((occa::kernelInfo*) info);
    occa::kernelArg_t &value_ = ((occaType_t*) value)->value;
    const int valueType       = ((occaType_t*) value)->type;

    switch(valueType){
    case OCCA_TYPE_INT    : info_.addDefine(macro, value_.int_);    break;
    case OCCA_TYPE_UINT   : info_.addDefine(macro, value_.uint_);   break;
    case OCCA_TYPE_CHAR   : info_.addDefine(macro, value_.char_);   break;
    case OCCA_TYPE_UCHAR  : info_.addDefine(macro, value_.uchar_);  break;
    case OCCA_TYPE_SHORT  : info_.addDefine(macro, value_.short_);  break;
    case OCCA_TYPE_USHORT : info_.addDefine(macro, value_.ushort_); break;
    case OCCA_TYPE_LONG   : info_.addDefine(macro, value_.long_);   break;
    case OCCA_TYPE_ULONG  : info_.addDefine(macro, value_.size_t_); break;

    case OCCA_TYPE_FLOAT  : info_.addDefine(macro, value_.float_);  break;
    case OCCA_TYPE_DOUBLE : info_.addDefine(macro, value_.double_); break;
    default:
      std::cout << "Wrong type input in [occaKernelInfoAddDefine]\n";
    }
  }

  void occaKernelInfoFree(occaKernelInfo info){
    delete (occa::kernelInfo*) info;
  }
  //====================================


  //---[ Memory ]-----------------------
  const char* occaMemoryMode(occaMemory memory){
    occa::memory &memory_ = *((occa::memory*) memory);

    return memory_.mode().c_str();
  }

  void occaCopyFromMem(occaMemory dest, occaMemory src,
                       const size_t bytes, const size_t offset){
    occa::memory &src_ = *((occa::memory*) src);
    occa::memory &dest_ = *((occa::memory*) dest);

    dest_.copyFrom(src_, bytes, offset);
  }

  void occaCopyFromPtr(occaMemory dest, void *src,
                       const size_t bytes, const size_t offset){
    occa::memory &dest_ = *((occa::memory*) dest);

    dest_.copyFrom(src, bytes, offset);
  }

  void occaCopyToMem(occaMemory dest, occaMemory src,
                     const size_t bytes, const size_t offset){
    occa::memory &src_ = *((occa::memory*) src);
    occa::memory &dest_ = *((occa::memory*) dest);

    src_.copyTo(dest_, bytes, offset);
  }

  void occaCopyToPtr(void *dest, occaMemory src,
                     const size_t bytes, const size_t offset){
    occa::memory &src_ = *((occa::memory*) src);

    src_.copyTo(dest, bytes, offset);
  }

  void occaAsyncCopyFromMem(occaMemory dest, occaMemory src,
                            const size_t bytes, const size_t offset){
    occa::memory &src_ = *((occa::memory*) src);
    occa::memory &dest_ = *((occa::memory*) dest);

    dest_.asyncCopyFrom(src_, bytes, offset);
  }

  void occaAsyncCopyFromPtr(occaMemory dest, void * src,
                            const size_t bytes, const size_t offset){
    occa::memory &dest_ = *((occa::memory*) dest);

    dest_.asyncCopyFrom(src, bytes, offset);
  }

  void occaAsyncCopyToMem(occaMemory dest, occaMemory src,
                          const size_t bytes, const size_t offset){
    occa::memory &src_ = *((occa::memory*) src);
    occa::memory &dest_ = *((occa::memory*) dest);

    src_.asyncCopyTo(dest_, bytes, offset);
  }

  void occaAsyncCopyToPtr(void *dest, occaMemory src,
                          const size_t bytes, const size_t offset){
    occa::memory &src_ = *((occa::memory*) src);

    src_.asyncCopyTo(src, bytes, offset);
  }

  void occaMemorySwap(occaMemory memoryA, occaMemory memoryB){
    occa::memory &memoryA_ = *((occa::memory*) memoryA);
    occa::memory &memoryB_ = *((occa::memory*) memoryB);

    memoryA_.swap(memoryB_);
  }


  void occaMemoryFree(occaMemory memory){
    occaMemory_t &memory_ = *((occaMemory_t*) memory);

    ((occa::memory*) memory_.ptr)->free();

    delete ((occa::memory*) memory_.ptr);

    delete &memory_;
  }
  //====================================

#  ifdef __cplusplus
}
#  endif
