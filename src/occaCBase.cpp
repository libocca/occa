#include "occaBase.hpp"

#define LIBOCCA_C_EXPORTS
#include "occaCBase.hpp"

#  ifdef __cplusplus
extern "C" {
#  endif

  // [-] Keep [int type] as the first entry
  struct occaMemory_t {
    int type;
    occa::memory mem;
  };

  // [-] Keep [int type] as the first entry
  struct occaType_t {
    int type;
    occa::kernelArg_t value;
  };

  struct occaArgumentList_t {
    int argc;
    occaMemory argv[100];
  };

  occaKernelInfo occaNoKernelInfo = NULL;

  const uintptr_t occaAutoSize = 0;
  const uintptr_t occaNoOffset = 0;

  const uintptr_t occaTypeSize[OCCA_TYPE_COUNT] = {
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
    sizeof(double),
    sizeof(char *)
  };

  //---[ TypeCasting ]------------------
  occaType LIBOCCA_CALLINGCONV occaInt(int value){
    occaType_t *type = new occaType_t;

    type->type       = OCCA_TYPE_INT;
    type->value.int_ = value;

    return (occaType) type;
  }

  occaType LIBOCCA_CALLINGCONV occaUInt(unsigned int value){
    occaType_t *type = new occaType_t;

    type->type        = OCCA_TYPE_UINT;
    type->value.uint_ = value;

    return (occaType) type;
  }

  occaType LIBOCCA_CALLINGCONV occaChar(char value){
    occaType_t *type = new occaType_t;

    type->type        = OCCA_TYPE_CHAR;
    type->value.char_ = value;

    return (occaType) type;
  }

  occaType LIBOCCA_CALLINGCONV occaUChar(unsigned char value){
    occaType_t *type = new occaType_t;

    type->type         = OCCA_TYPE_UCHAR;
    type->value.uchar_ = value;

    return (occaType) type;
  }

  occaType LIBOCCA_CALLINGCONV occaShort(short value){
    occaType_t *type = new occaType_t;

    type->type         = OCCA_TYPE_SHORT;
    type->value.short_ = value;

    return (occaType) type;
  }

  occaType LIBOCCA_CALLINGCONV occaUShort(unsigned short value){
    occaType_t *type = new occaType_t;

    type->type          = OCCA_TYPE_USHORT;
    type->value.ushort_ = value;

    return (occaType) type;
  }

  occaType LIBOCCA_CALLINGCONV occaLong(long value){
    occaType_t *type = new occaType_t;

    type->type        = OCCA_TYPE_LONG;
    type->value.long_ = value;

    return (occaType) type;
  }

  occaType LIBOCCA_CALLINGCONV occaULong(unsigned long value){
    occaType_t *type = new occaType_t;

    type->type          = OCCA_TYPE_ULONG;
    type->value.uintptr_t_ = value;

    return (occaType) type;
  }

  occaType LIBOCCA_CALLINGCONV occaFloat(float value){
    occaType_t *type = new occaType_t;

    type->type         = OCCA_TYPE_FLOAT;
    type->value.float_ = value;

    return (occaType) type;
  }

  occaType LIBOCCA_CALLINGCONV occaDouble(double value){
    occaType_t *type = new occaType_t;

    type->type          = OCCA_TYPE_DOUBLE;
    type->value.double_ = value;

    return (occaType) type;
  }

  occaType LIBOCCA_CALLINGCONV occaString(char *value){
    occaType_t *type = new occaType_t;

    type->type        = OCCA_TYPE_STRING;
    type->value.void_ = value;

    return (occaType) type;
  }
  //====================================


  //---[ Device ]-----------------------
  const char* LIBOCCA_CALLINGCONV occaDeviceMode(occaDevice device){
    occa::device &device_ = *((occa::device*) device);

    return device_.mode().c_str();
  }

  void LIBOCCA_CALLINGCONV occaDeviceSetCompiler(occaDevice device,
                                                  const char *compiler){
    occa::device &device_ = *((occa::device*) device);
    device_.setCompiler(compiler);
  }

  void LIBOCCA_CALLINGCONV occaDeviceSetCompilerFlags(occaDevice device,
                                                       const char *compilerFlags){
    occa::device &device_ = *((occa::device*) device);
    device_.setCompilerFlags(compilerFlags);
  }

  occaDevice LIBOCCA_CALLINGCONV occaGetDevice(const char *infos){
    occa::device *device = new occa::device();

    device->setup(infos);

    return (occaDevice) device;
  }

  occaDevice LIBOCCA_CALLINGCONV occaGetDeviceFromArgs(const char *mode,
                                                       int arg1, int arg2){
    occa::device *device = new occa::device();

    device->setup(mode, arg1, arg2);

    return (occaDevice) device;
  }

  occaKernel LIBOCCA_CALLINGCONV occaBuildKernelFromSource(occaDevice device,
                                                            const char *filename,
                                                            const char *functionName,
                                                            occaKernelInfo info){
    occa::device &device_  = *((occa::device*) device);

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

  occaKernel LIBOCCA_CALLINGCONV occaBuildKernelFromBinary(occaDevice device,
                                                            const char *filename,
                                                            const char *functionName){
    occa::device &device_ = *((occa::device*) device);

    occa::kernel *kernel = new occa::kernel();

    *kernel = device_.buildKernelFromBinary(filename, functionName);

    return (occaKernel) kernel;
  }

  occaKernel LIBOCCA_CALLINGCONV occaBuildKernelFromLoopy(occaDevice device,
                                                           const char *filename,
                                                           const char *functionName,
                                                           occaKernelInfo info){
    occa::device &device_  = *((occa::device*) device);

    occa::kernel *kernel = new occa::kernel();

    occa::kernelInfo &info_ = *((occa::kernelInfo*) info);

    *kernel = device_.buildKernelFromLoopy(filename,
                                           functionName,
                                           info_,
                                           occa::useLoopy);

    return (occaKernel) kernel;
  }

  occaKernel LIBOCCA_CALLINGCONV occaBuildKernelFromFloopy(occaDevice device,
                                                           const char *filename,
                                                           const char *functionName,
                                                           occaKernelInfo info){
    occa::device &device_  = *((occa::device*) device);

    occa::kernel *kernel = new occa::kernel();

    occa::kernelInfo &info_ = *((occa::kernelInfo*) info);

    *kernel = device_.buildKernelFromLoopy(filename,
                                           functionName,
                                           info_,
                                           occa::useFloopy);

    return (occaKernel) kernel;
  }

  occaMemory LIBOCCA_CALLINGCONV occaDeviceMalloc(occaDevice device,
                                                   uintptr_t bytes,
                                                   void *source){
    occa::device &device_ = *((occa::device*) device);

    occaMemory_t *memory = new occaMemory_t();

    memory->type = OCCA_TYPE_MEMORY;
    memory->mem = device_.malloc(bytes, source);

    return (occaMemory) memory;
  }

  void LIBOCCA_CALLINGCONV occaDeviceFlush(occaDevice device){
    occa::device &device_ = *((occa::device*) device);

    device_.flush();
  }

  void LIBOCCA_CALLINGCONV occaDeviceFinish(occaDevice device){
    occa::device &device_ = *((occa::device*) device);

    device_.finish();
  }

  occaStream LIBOCCA_CALLINGCONV occaDeviceGenStream(occaDevice device){
    occa::device &device_ = *((occa::device*) device);

    return (occaStream) device_.createStream();
  }

  occaStream LIBOCCA_CALLINGCONV occaDeviceGetStream(occaDevice device){
    occa::device &device_ = *((occa::device*) device);

    return (occaStream) device_.getStream();
  }

  void LIBOCCA_CALLINGCONV occaDeviceSetStream(occaDevice device, occaStream stream){
    occa::device &device_ = *((occa::device*) device);
    occa::stream &stream_ = *((occa::stream*) stream);

    device_.setStream(stream_);
  }

  occaTag LIBOCCA_CALLINGCONV occaDeviceTagStream(occaDevice device){
    occa::device &device_ = *((occa::device*) device);

    occa::tag oldTag = device_.tagStream();
    occaTag newTag;

    ::memcpy(&newTag, &oldTag, sizeof(oldTag));

    return newTag;
  }

  double LIBOCCA_CALLINGCONV occaDeviceTimeBetweenTags(occaDevice device,
                                                        occaTag startTag, occaTag endTag){
    occa::device &device_ = *((occa::device*) device);

    occa::tag startTag_, endTag_;

    ::memcpy(&startTag_, &startTag, sizeof(startTag_));
    ::memcpy(&endTag_  , &endTag  , sizeof(endTag_));

    return device_.timeBetween(startTag_, endTag_);
  }

  void LIBOCCA_CALLINGCONV occaDeviceStreamFree(occaDevice device, occaStream stream){
    occa::device &device_ = *((occa::device*) device);
    occa::stream &stream_ = *((occa::stream*) stream);

    device_.free(stream_);
  }

  void LIBOCCA_CALLINGCONV occaDeviceFree(occaDevice device){
    occa::device &device_ = *((occa::device*) device);

    device_.free();

    delete (occa::device*) device;
  }
  //====================================


  //---[ Kernel ]-----------------------
  occaDim LIBOCCA_CALLINGCONV occaGenDim(uintptr_t x, uintptr_t y, uintptr_t z){
    occaDim ret;

    ret.x = x;
    ret.y = y;
    ret.z = z;

    return ret;
  }

  const char* LIBOCCA_CALLINGCONV occaKernelMode(occaKernel kernel){
    occa::kernel &kernel_ = *((occa::kernel*) kernel);

    return kernel_.mode().c_str();
  }

  int LIBOCCA_CALLINGCONV occaKernelPreferredDimSize(occaKernel kernel){
    occa::kernel &kernel_ = *((occa::kernel*) kernel);

    return kernel_.preferredDimSize();
  }

  void LIBOCCA_CALLINGCONV occaKernelSetWorkingDims(occaKernel kernel,
                                                     int dims,
                                                     occaDim items,
                                                     occaDim groups){
    occa::kernel &kernel_ = *((occa::kernel*) kernel);

    kernel_.setWorkingDims(dims,
                           occa::dim(items.x, items.y, items.z),
                           occa::dim(groups.x, groups.y, groups.z));
  }

  void LIBOCCA_CALLINGCONV occaKernelSetAllWorkingDims(occaKernel kernel,
                                                        int dims,
                                                        uintptr_t itemsX, uintptr_t itemsY, uintptr_t itemsZ,
                                                        uintptr_t groupsX, uintptr_t groupsY, uintptr_t groupsZ){
    occa::kernel &kernel_ = *((occa::kernel*) kernel);

    kernel_.setWorkingDims(dims,
                           occa::dim(itemsX, itemsY, itemsZ),
                           occa::dim(groupsX, groupsY, groupsZ));
  }


  double LIBOCCA_CALLINGCONV occaKernelTimeTaken(occaKernel kernel){
    occa::kernel &kernel_ = *((occa::kernel*) kernel);

    return kernel_.timeTaken();
  }

  occaArgumentList LIBOCCA_CALLINGCONV occaGenArgumentList(){
    occaArgumentList_t *list = new occaArgumentList_t();
    list->argc = 0;

    return (occaArgumentList) list;
  }

  void LIBOCCA_CALLINGCONV occaArgumentListClear(occaArgumentList list){
    occaArgumentList_t &list_ = *((occaArgumentList_t*) list);

    for(int i = 0; i < list_.argc; ++i){
      occaType_t &type_ = *((occaType_t*) list_.argv[i]);

      if(type_.type != OCCA_TYPE_MEMORY)
        delete (occaType_t*) list_.argv[i];
    }

    list_.argc = 0;
  }

  void LIBOCCA_CALLINGCONV occaArgumentListFree(occaArgumentList list){
    delete (occaArgumentList_t*) list;
  }

  void LIBOCCA_CALLINGCONV occaArgumentListAddArg(occaArgumentList list,
                                                   int argPos,
                                                   void * type){
    occaArgumentList_t &list_ = *((occaArgumentList_t*) list);

    if(list_.argc < (argPos + 1)){
      OCCA_CHECK(argPos < OCCA_MAX_ARGS);

      list_.argc = (argPos + 1);
    }

    list_.argv[argPos] = (occaMemory_t*) type;
  }

  // Note the _
  //   Macro that is called > API function that is never seen
  void LIBOCCA_CALLINGCONV occaKernelRun_(occaKernel kernel,
                                           occaArgumentList list){
    occa::kernel &kernel_     = *((occa::kernel*) kernel);
    occaArgumentList_t &list_ = *((occaArgumentList_t*) list);

    kernel_.clearArgumentList();

    for(int i = 0; i < list_.argc; ++i){
      occaMemory_t &memory_ = *((occaMemory_t*) list_.argv[i]);

      if(memory_.type == OCCA_TYPE_MEMORY){
        kernel_.addArgument(i, occa::kernelArg(memory_.mem));
      }
      else{
        occaType_t &type_ = *((occaType_t*) list_.argv[i]);

        kernel_.addArgument(i, occa::kernelArg(type_.value,
                                               occaTypeSize[type_.type],
                                               false));
      }
    }

    kernel_.runFromArguments();
  }

  OCCA_C_KERNEL_RUN_DEFINITIONS;

  void LIBOCCA_CALLINGCONV occaKernelFree(occaKernel kernel){
    occa::kernel &kernel_ = *((occa::kernel*) kernel);

    kernel_.free();

    delete (occa::kernel*) kernel;
  }

  occaKernelInfo LIBOCCA_CALLINGCONV occaGenKernelInfo(){
    occa::kernelInfo *info = new occa::kernelInfo();

    return (occaKernelInfo) info;

  }

  void LIBOCCA_CALLINGCONV occaKernelInfoAddDefine(occaKernelInfo info,
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
    case OCCA_TYPE_ULONG  : info_.addDefine(macro, value_.uintptr_t_); break;

    case OCCA_TYPE_FLOAT  : info_.addDefine(macro, value_.float_);  break;
    case OCCA_TYPE_DOUBLE : info_.addDefine(macro, value_.double_); break;

    case OCCA_TYPE_STRING : info_.addDefine(macro, std::string((char*) value_.void_)); break;
    default:
      std::cout << "Wrong type input in [occaKernelInfoAddDefine]\n";
    }
  }

  void LIBOCCA_CALLINGCONV occaKernelInfoFree(occaKernelInfo info){
    delete (occa::kernelInfo*) info;
  }
  //====================================


  //---[ Wrappers ]---------------------
#if OCCA_OPENCL_ENABLED
  LIBOCCA_API occaDevice LIBOCCA_CALLINGCONV occaWrapOpenCLDevice(cl_platform_id platformID,
                                                                  cl_device_id deviceID,
                                                                  cl_context context){
    occa::device *device = new occa::device();

    *device = occa::cl::wrapDevice(platformID, deviceID, context);

    return (occaDevice) device;
  }
#endif

#if OCCA_CUDA_ENABLED
  LIBOCCA_API occaDevice LIBOCCA_CALLINGCONV occaWrapCudaDevice(CUdevice device, CUcontext context){
    occa::device *device = new occa::device();

    *device = occa::cu::wrapDevice(device, context);

    return (occaDevice) device;
  }
#endif

#if OCCA_COI_ENABLED
  LIBOCCA_API occaDevice LIBOCCA_CALLINGCONV occaWrapCoiDevice(COIENGINE coiDevice){
    occa::device *device = new occa::device();

    *device = occa::coi::wrapDevice(coiDevice);

    return (occaDevice) device;
  }
#endif

  LIBOCCA_API occaMemory LIBOCCA_CALLINGCONV occaDeviceWrapMemory(occaDevice device,
                                                                  void *handle_,
                                                                  const uintptr_t bytes){
    occa::device &device_ = *((occa::device*) device);

    occaMemory_t *memory = new occaMemory_t();

    memory->type = OCCA_TYPE_MEMORY;
    memory->mem = device_.wrapMemory(handle_, bytes);

    return (occaMemory) memory;
  }

  LIBOCCA_API occaStream LIBOCCA_CALLINGCONV occaDeviceWrapStream(occaDevice device, void *handle_){
    occa::device &device_ = *((occa::device*) device);
    occaStream *stream = new occaStream;

    *stream = device_.wrapStream(handle_);

    return (occaStream) stream;
  }
  //====================================


  //---[ Memory ]-----------------------
  const char* LIBOCCA_CALLINGCONV occaMemoryMode(occaMemory memory){
    occa::memory &memory_ = memory->mem;

    return memory_.mode().c_str();
  }

  void LIBOCCA_CALLINGCONV occaCopyMemToMem(occaMemory dest, occaMemory src,
                                             const uintptr_t bytes,
                                             const uintptr_t destOffset,
                                             const uintptr_t srcOffset){
    occa::memory &src_  = src->mem;
    occa::memory &dest_ = dest->mem;

    memcpy(dest_, src_, bytes, destOffset, srcOffset);
  }

  void LIBOCCA_CALLINGCONV occaCopyPtrToMem(occaMemory dest, const void *src,
                                             const uintptr_t bytes,
                                             const uintptr_t offset){
    occa::memory &dest_ = dest->mem;

    memcpy(dest_, src, bytes, offset);
  }

  void LIBOCCA_CALLINGCONV occaCopyMemToPtr(void *dest, occaMemory src,
                                             const uintptr_t bytes,
                                             const uintptr_t offset){
    occa::memory &src_ = src->mem;

    memcpy(dest, src_, bytes, offset);
  }

  void LIBOCCA_CALLINGCONV occaAsyncCopyMemToMem(occaMemory dest, occaMemory src,
                                                  const uintptr_t bytes,
                                                  const uintptr_t destOffset,
                                                  const uintptr_t srcOffset){
    occa::memory &src_  = src->mem;
    occa::memory &dest_ = dest->mem;

    asyncMemcpy(dest_, src_, bytes, destOffset, srcOffset);
  }

  void LIBOCCA_CALLINGCONV occaAsyncCopyPtrToMem(occaMemory dest, const void * src,
                                                  const uintptr_t bytes,
                                                  const uintptr_t offset){
    occa::memory &dest_ = dest->mem;

    asyncMemcpy(dest_, src, bytes, offset);
  }

  void LIBOCCA_CALLINGCONV occaAsyncCopyMemToPtr(void *dest, occaMemory src,
                                                  const uintptr_t bytes,
                                                  const uintptr_t offset){
    occa::memory &src_ = src->mem;

    asyncMemcpy(dest, src_, bytes, offset);
  }

  void LIBOCCA_CALLINGCONV occaMemorySwap(occaMemory memoryA, occaMemory memoryB){
    occa::memory &memoryA_ = memoryA->mem;
    occa::memory &memoryB_ = memoryB->mem;

    memoryA_.swap(memoryB_);
  }


  void LIBOCCA_CALLINGCONV occaMemoryFree(occaMemory memory){
    memory->mem.free();

    delete memory;
  }
  //====================================

#  ifdef __cplusplus
}
#  endif
