#include "occa/base.hpp"

#define OCCA_C_EXPORTS

#include "occa/cBase.hpp"

OCCA_START_EXTERN_C

struct occaType_t {
  int type;
  occa::kernelArg_t value;

  inline occaType_t() :
    type(),
    value() {}

  inline occaType_t(int type_) :
    type(type_),
    value() {}
};

struct occaTypePtr_t {
  struct occaType_t *ptr;

  inline occaTypePtr_t() :
    ptr(new occaType_t()) {}

  inline occaTypePtr_t(void *ptr_) :
    ptr((occaType_t*) ptr_) {}

  inline occaTypePtr_t(int type_) :
    ptr(new occaType_t(type_)) {}

  inline occaType_t& occaType() {
    return *ptr;
  }

  inline int& type() {
    return ptr->type;
  }

  inline int type() const  {
    return ptr->type;
  }

  inline occa::kernelArg_t& value() {
    return ptr->value;
  }

  inline occa::kernelArg_t value() const  {
    return ptr->value;
  }

  inline void swap(occaTypePtr_t *tp) {
    occaType_t *tmp = ptr;
    ptr = tp->ptr;
    tp->ptr = tmp;
  }
};

struct occaArgumentList_t {
  int argc;
  occaType_t *argv[100];
};

//---[ Globals & Flags ]------------
occaKernelInfo occaNoKernelInfo = NULL;

const uintptr_t occaAutoSize = 0;
const uintptr_t occaNoOffset = 0;

const int occaUsingOKL    = occa::usingOKL;
const int occaUsingOFL    = occa::usingOFL;
const int occaUsingNative = occa::usingNative;

void OCCA_RFUNC occaSetVerboseCompilation(const int value) {
  occa::setVerboseCompilation((bool) value);
}
//==================================

OCCA_END_EXTERN_C


//---[ TypeCasting ]------------------
namespace occa {
  std::string typeToStr(occaType value) {
    occa::kernelArg_t &value_ = value->value();
    const int valueType       = value->type();

    switch(valueType) {
    case OCCA_TYPE_INT    : return occa::toString(value_.data.int_);
    case OCCA_TYPE_UINT   : return occa::toString(value_.data.uint_);
    case OCCA_TYPE_CHAR   : return occa::toString(value_.data.char_);
    case OCCA_TYPE_UCHAR  : return occa::toString(value_.data.uchar_);
    case OCCA_TYPE_SHORT  : return occa::toString(value_.data.short_);
    case OCCA_TYPE_USHORT : return occa::toString(value_.data.ushort_);
    case OCCA_TYPE_LONG   : return occa::toString(value_.data.long_);
    case OCCA_TYPE_ULONG  : return occa::toString(value_.data.uintptr_t_);

    case OCCA_TYPE_FLOAT  : return occa::toString(value_.data.float_);
    case OCCA_TYPE_DOUBLE : return occa::toString(value_.data.double_);

    case OCCA_TYPE_STRING : return std::string((char*) value_.data.void_);
    default:
      std::cout << "Wrong type input in [occaKernelInfoAddDefine]\n";
    }

    return "";
  }
}

OCCA_START_EXTERN_C

occaType OCCA_RFUNC occaPtr(void *ptr) {
  occaType type = (occaType) new occaTypePtr_t(OCCA_TYPE_INT);
  type->value().data.void_ = ptr;
  return type;
}

occaType OCCA_RFUNC occaInt(int value) {
  occaType type = (occaType) new occaTypePtr_t(OCCA_TYPE_INT);
  type->value().size      = sizeof(int);
  type->value().data.int_ = value;
  type->value().info      = occa::kArgInfo::none;
  return type;
}

occaType OCCA_RFUNC occaUInt(unsigned int value) {
  occaType type = (occaType) new occaTypePtr_t(OCCA_TYPE_UINT);
  type->value().size       = sizeof(unsigned int);
  type->value().data.uint_ = value;
  type->value().info       = occa::kArgInfo::none;
  return type;
}

occaType OCCA_RFUNC occaChar(char value) {
  occaType type = (occaType) new occaTypePtr_t(OCCA_TYPE_CHAR);
  type->value().size       = sizeof(char);
  type->value().data.char_ = value;
  type->value().info       = occa::kArgInfo::none;
  return type;
}

occaType OCCA_RFUNC occaUChar(unsigned char value) {
  occaType type = (occaType) new occaTypePtr_t(OCCA_TYPE_UCHAR);
  type->value().size        = sizeof(unsigned char);
  type->value().data.uchar_ = value;
  type->value().info        = occa::kArgInfo::none;
  return type;
}

occaType OCCA_RFUNC occaShort(short value) {
  occaType type = (occaType) new occaTypePtr_t(OCCA_TYPE_SHORT);
  type->value().size        = sizeof(short);
  type->value().data.short_ = value;
  type->value().info        = occa::kArgInfo::none;
  return type;
}

occaType OCCA_RFUNC occaUShort(unsigned short value) {
  occaType type = (occaType) new occaTypePtr_t(OCCA_TYPE_USHORT);
  type->value().size         = sizeof(unsigned short);
  type->value().data.ushort_ = value;
  type->value().info         = occa::kArgInfo::none;
  return type;
}

occaType OCCA_RFUNC occaLong(long value) {
  occaType type = (occaType) new occaTypePtr_t(OCCA_TYPE_LONG);
  type->value().size       = sizeof(long);
  type->value().data.long_ = value;
  type->value().info       = occa::kArgInfo::none;
  return type;
}

occaType OCCA_RFUNC occaULong(unsigned long value) {
  occaType type = (occaType) new occaTypePtr_t(OCCA_TYPE_ULONG);
  type->value().size            = sizeof(unsigned long);
  type->value().data.uintptr_t_ = value;
  type->value().info            = occa::kArgInfo::none;
  return type;
}

occaType OCCA_RFUNC occaFloat(float value) {
  occaType type = (occaType) new occaTypePtr_t(OCCA_TYPE_FLOAT);
  type->value().size        = sizeof(float);
  type->value().data.float_ = value;
  type->value().info        = occa::kArgInfo::none;
  return type;
}

occaType OCCA_RFUNC occaDouble(double value) {
  occaType type = (occaType) new occaTypePtr_t(OCCA_TYPE_DOUBLE);
  type->value().size         = sizeof(double);
  type->value().data.double_ = value;
  type->value().info         = occa::kArgInfo::none;
  return type;
}

occaType OCCA_RFUNC occaStruct(void *value, uintptr_t bytes) {
  occaType type = (occaType) new occaTypePtr_t(OCCA_TYPE_STRUCT);
  type->value().size       = bytes;
  type->value().data.void_ = value;
  type->value().info       = occa::kArgInfo::usePointer;
  return type;
}

occaType OCCA_RFUNC occaString(const char *value) {
  occaType type = (occaType) new occaTypePtr_t(OCCA_TYPE_STRING);
  type->value().size       = sizeof(char*);
  type->value().data.void_ = const_cast<char*>(value);
  type->value().info       = occa::kArgInfo::usePointer;
  return type;
}
//====================================


//---[ Background Device ]------------
//  |---[ Device ]--------------------
void OCCA_RFUNC occaSetDevice(occaDevice device) {
  occa::device device_((occa::device_v*) device);

  occa::setDevice(device_);
}

void OCCA_RFUNC occaSetDeviceFromInfo(const char *infos) {
  occa::setDevice(infos);
}

occaDevice OCCA_RFUNC occaGetCurrentDevice() {
  occa::device device = occa::getCurrentDevice();
  return (occaDevice) device.getDHandle();
}

void OCCA_RFUNC occaSetCompiler(const char *compiler_) {
  occa::setCompiler(compiler_);
}

void OCCA_RFUNC occaSetCompilerEnvScript(const char *compilerEnvScript_) {
  occa::setCompilerEnvScript(compilerEnvScript_);
}

void OCCA_RFUNC occaSetCompilerFlags(const char *compilerFlags_) {
  occa::setCompilerFlags(compilerFlags_);
}

const char* OCCA_RFUNC occaGetCompiler() {
  return occa::getCompiler().c_str();
}

const char* OCCA_RFUNC occaGetCompilerEnvScript() {
  return occa::getCompilerEnvScript().c_str();
}

const char* OCCA_RFUNC occaGetCompilerFlags() {
  return occa::getCompilerFlags().c_str();
}

void OCCA_RFUNC occaFlush() {
  occa::flush();
}

void OCCA_RFUNC occaFinish() {
  occa::finish();
}

void OCCA_RFUNC occaWaitFor(occaStreamTag tag) {
  occa::streamTag tag_;

  ::memcpy(&tag_, &tag, sizeof(tag_));

  occa::waitFor(tag_);
}

occaStream OCCA_RFUNC occaCreateStream() {
  occa::stream &newStream = *(new occa::stream(occa::createStream()));

  return (occaStream) &newStream;
}

occaStream OCCA_RFUNC occaGetStream() {
  occa::stream &currentStream = *(new occa::stream(occa::getStream()));

  return (occaStream) &currentStream;
}

void OCCA_RFUNC occaSetStream(occaStream stream) {
  occa::setStream(*((occa::stream*) stream));
}

occaStream OCCA_RFUNC occaWrapStream(void *handle_) {
  occa::stream &newStream = *(new occa::stream(occa::wrapStream(handle_)));

  return (occaStream) &newStream;
}

occaStreamTag OCCA_RFUNC occaTagStream() {
  occa::streamTag oldTag = occa::tagStream();
  occaStreamTag newTag;

  newTag.tagTime = oldTag.tagTime;

  ::memcpy(&(newTag.handle), &(oldTag.handle), sizeof(void*));

  return newTag;
}
//  |=================================

//  |---[ Kernel ]--------------------
occaKernel OCCA_RFUNC occaBuildKernel(const char *str,
                                      const char *functionName,
                                      occaKernelInfo info) {
  occa::kernel kernel;

  if(info != occaNoKernelInfo) {
    occa::kernelInfo &info_ = *((occa::kernelInfo*) info);

    kernel = occa::buildKernel(str,
                               functionName,
                               info_);
  }
  else{
    kernel = occa::buildKernel(str,
                               functionName);
  }

  return (occaKernel) kernel.getKHandle();
}

occaKernel OCCA_RFUNC occaBuildKernelFromSource(const char *filename,
                                                const char *functionName,
                                                occaKernelInfo info) {
  occa::kernel kernel;

  if(info != occaNoKernelInfo) {
    occa::kernelInfo &info_ = *((occa::kernelInfo*) info);

    kernel = occa::buildKernelFromSource(filename,
                                         functionName,
                                         info_);
  }
  else{
    kernel = occa::buildKernelFromSource(filename,
                                         functionName);
  }

  return (occaKernel) kernel.getKHandle();
}

occaKernel OCCA_RFUNC occaBuildKernelFromString(const char *str,
                                                const char *functionName,
                                                occaKernelInfo info,
                                                const int language) {
  occa::kernel kernel;

  if(info != occaNoKernelInfo) {
    occa::kernelInfo &info_ = *((occa::kernelInfo*) info);

    kernel = occa::buildKernelFromString(str,
                                         functionName,
                                         info_,
                                         language);
  }
  else{
    kernel = occa::buildKernelFromString(str,
                                         functionName,
                                         language);
  }

  return (occaKernel) kernel.getKHandle();
}

occaKernel OCCA_RFUNC occaBuildKernelFromBinary(const char *filename,
                                                const char *functionName) {
  occa::kernel kernel;

  kernel = occa::buildKernelFromBinary(filename, functionName);

  return (occaKernel) kernel.getKHandle();
}
//  |=================================

//  |---[ Memory ]--------------------
void OCCA_RFUNC occaMemorySwap(occaMemory a, occaMemory b) {
  a->swap(b);
}

occaMemory OCCA_RFUNC occaWrapMemory(void *handle_,
                                     const uintptr_t bytes) {

  occa::memory memory_ = occa::wrapMemory(handle_, bytes);

  occaMemory memory = (occaMemory) new occaTypePtr_t(OCCA_TYPE_MEMORY);
  memory->value().data.void_ = memory_.getMHandle();
  return memory;
}

void OCCA_RFUNC occaWrapManagedMemory(void *handle_,
                                      const uintptr_t bytes) {
  occa::wrapManagedMemory(handle_, bytes);
}

occaMemory OCCA_RFUNC occaMalloc(const uintptr_t bytes,
                                 void *src) {
  occa::memory memory_ = occa::malloc(bytes, src);

  occaMemory memory = (occaMemory) new occaTypePtr_t();
  memory->type()             = OCCA_TYPE_MEMORY;
  memory->value().data.void_ = memory_.getMHandle();
  return memory;
}

void* OCCA_RFUNC occaManagedAlloc(const uintptr_t bytes,
                                  void *src) {

  return occa::managedAlloc(bytes, src);
}

occaMemory OCCA_RFUNC occaMappedAlloc(const uintptr_t bytes,
                                      void *src) {

  occa::memory memory_ = occa::mappedAlloc(bytes, src);

  occaMemory memory = (occaMemory) new occaTypePtr_t();
  memory->type()             = OCCA_TYPE_MEMORY;
  memory->value().data.void_ = memory_.getMHandle();
  return memory;
}

void* OCCA_RFUNC occaManagedMappedAlloc(const uintptr_t bytes,
                                        void *src) {

  return occa::managedMappedAlloc(bytes, src);
}
//  |=================================
//====================================


//---[ Device ]-----------------------
void OCCA_RFUNC occaPrintAvailableDevices() {
  occa::printAvailableDevices();
}

occaDeviceInfo OCCA_RFUNC occaCreateDeviceInfo() {
  occa::deviceInfo *info = new occa::deviceInfo();

  return (occaDeviceInfo) info;
}

void OCCA_RFUNC occaDeviceInfoAppend(occaDeviceInfo info,
                                     const char *key,
                                     const char *value) {

  occa::deviceInfo &info_ = *((occa::deviceInfo*) info);

  info_.append(key, value);
}

void OCCA_RFUNC occaDeviceInfoAppendType(occaDeviceInfo info,
                                         const char *key,
                                         occaType value) {

  occa::deviceInfo &info_ = *((occa::deviceInfo*) info);

  info_.append(key, occa::typeToStr(value));

  delete value;
}

void OCCA_RFUNC occaDeviceInfoFree(occaDeviceInfo info) {
  delete (occa::deviceInfo*) info;
}

occaDevice OCCA_RFUNC occaCreateDevice(const char *infos) {
  occa::device device(infos);

  return (occaDevice) device.getDHandle();
}

occaDevice OCCA_RFUNC occaCreateDeviceFromInfo(occaDeviceInfo dInfo) {
  occa::device device(*((occa::deviceInfo*) dInfo));

  return (occaDevice) device.getDHandle();
}

occaDevice OCCA_RFUNC occaCreateDeviceFromArgs(const char *mode,
                                               int arg1, int arg2) {
  occa::device device;
  device.setup(mode, arg1, arg2);

  return (occaDevice) device.getDHandle();
}

const char* OCCA_RFUNC occaDeviceMode(occaDevice device) {
  occa::device device_((occa::device_v*) device);

  return device_.mode().c_str();
}

void OCCA_RFUNC occaDeviceSetCompiler(occaDevice device,
                                      const char *compiler) {
  occa::device device_((occa::device_v*) device);

  device_.setCompiler(compiler);
}

void OCCA_RFUNC occaDeviceSetCompilerEnvScript(occaDevice device,
                                               const char *compilerEnvScript_) {
  occa::device device_((occa::device_v*) device);

  device_.setCompilerEnvScript(compilerEnvScript_);
}

void OCCA_RFUNC occaDeviceSetCompilerFlags(occaDevice device,
                                           const char *compilerFlags) {
  occa::device device_((occa::device_v*) device);

  device_.setCompilerFlags(compilerFlags);
}

const char* OCCA_RFUNC occaDeviceGetCompiler(occaDevice device) {
  occa::device device_((occa::device_v*) device);
  return device_.getCompiler().c_str();
}

const char* OCCA_RFUNC occaDeviceGetCompilerEnvScript(occaDevice device) {
  occa::device device_((occa::device_v*) device);
  return device_.getCompilerFlags().c_str();
}

const char* OCCA_RFUNC occaDeviceGetCompilerFlags(occaDevice device) {
  occa::device device_((occa::device_v*) device);
  return device_.getCompilerEnvScript().c_str();
}

uintptr_t OCCA_RFUNC occaDeviceMemorySize(occaDevice device) {
  occa::device device_((occa::device_v*) device);

  return device_.memorySize();
}

uintptr_t OCCA_RFUNC occaDeviceMemoryAllocated(occaDevice device) {
  occa::device device_((occa::device_v*) device);

  return device_.memoryAllocated();
}

// Old version of [occaDeviceMemoryAllocated()]
uintptr_t OCCA_RFUNC occaDeviceBytesAllocated(occaDevice device) {
  occa::device device_((occa::device_v*) device);

  return device_.memoryAllocated();
}

occaKernel OCCA_RFUNC occaDeviceBuildKernel(occaDevice device,
                                            const char *str,
                                            const char *functionName,
                                            occaKernelInfo info) {
  occa::device device_((occa::device_v*) device);
  occa::kernel kernel;

  if(info != occaNoKernelInfo) {
    occa::kernelInfo &info_ = *((occa::kernelInfo*) info);

    kernel = device_.buildKernel(str,
                                 functionName,
                                 info_);
  }
  else{
    kernel = device_.buildKernel(str,
                                 functionName);
  }

  return (occaKernel) kernel.getKHandle();
}

occaKernel OCCA_RFUNC occaDeviceBuildKernelFromSource(occaDevice device,
                                                      const char *filename,
                                                      const char *functionName,
                                                      occaKernelInfo info) {
  occa::device device_((occa::device_v*) device);
  occa::kernel kernel;

  if(info != occaNoKernelInfo) {
    occa::kernelInfo &info_ = *((occa::kernelInfo*) info);

    kernel = device_.buildKernelFromSource(filename,
                                           functionName,
                                           info_);
  }
  else{
    kernel = device_.buildKernelFromSource(filename,
                                           functionName);
  }

  return (occaKernel) kernel.getKHandle();
}

occaKernel OCCA_RFUNC occaDeviceBuildKernelFromString(occaDevice device,
                                                      const char *str,
                                                      const char *functionName,
                                                      occaKernelInfo info,
                                                      const int language) {
  occa::device device_((occa::device_v*) device);
  occa::kernel kernel;

  if(info != occaNoKernelInfo) {
    occa::kernelInfo &info_ = *((occa::kernelInfo*) info);

    kernel = device_.buildKernelFromString(str,
                                           functionName,
                                           info_,
                                           language);
  }
  else{
    kernel = device_.buildKernelFromString(str,
                                           functionName,
                                           language);
  }

  return (occaKernel) kernel.getKHandle();
}

occaKernel OCCA_RFUNC occaDeviceBuildKernelFromBinary(occaDevice device,
                                                      const char *filename,
                                                      const char *functionName) {
  occa::device device_((occa::device_v*) device);
  occa::kernel kernel;

  kernel = device_.buildKernelFromBinary(filename, functionName);

  return (occaKernel) kernel.getKHandle();
}

occaMemory OCCA_RFUNC occaDeviceMalloc(occaDevice device,
                                       uintptr_t bytes,
                                       void *src) {

  occa::device device_((occa::device_v*) device);
  occa::memory memory_ = device_.malloc(bytes, src);

  occaMemory memory = (occaMemory) new occaTypePtr_t();
  memory->type()             = OCCA_TYPE_MEMORY;
  memory->value().data.void_ = memory_.getMHandle();
  return memory;
}

void* OCCA_RFUNC occaDeviceManagedAlloc(occaDevice device,
                                        uintptr_t bytes,
                                        void *src) {

  occa::device device_((occa::device_v*) device);

  return device_.managedAlloc(bytes, src);
}

occaMemory OCCA_RFUNC occaDeviceMappedAlloc(occaDevice device,
                                            uintptr_t bytes,
                                            void *src) {

  occa::device device_((occa::device_v*) device);
  occa::memory memory_ = device_.mappedAlloc(bytes, src);

  occaMemory memory = (occaMemory) new occaTypePtr_t();
  memory->type()             = OCCA_TYPE_MEMORY;
  memory->value().data.void_ = memory_.getMHandle();
  return memory;
}

void* OCCA_RFUNC occaDeviceManagedMappedAlloc(occaDevice device,
                                              uintptr_t bytes,
                                              void *src) {

  occa::device device_((occa::device_v*) device);

  return device_.managedMappedAlloc(bytes, src);
}

void OCCA_RFUNC occaDeviceFlush(occaDevice device) {
  occa::device device_((occa::device_v*) device);

  device_.flush();
}

void OCCA_RFUNC occaDeviceFinish(occaDevice device) {
  occa::device device_((occa::device_v*) device);

  device_.finish();
}

occaStream OCCA_RFUNC occaDeviceCreateStream(occaDevice device) {
  occa::device device_((occa::device_v*) device);

  occa::stream &newStream = *(new occa::stream(device_.createStream()));

  return (occaStream) &newStream;
}

occaStream OCCA_RFUNC occaDeviceGetStream(occaDevice device) {
  occa::device device_((occa::device_v*) device);

  occa::stream &currentStream = *(new occa::stream(device_.getStream()));

  return (occaStream) &currentStream;
}

void OCCA_RFUNC occaDeviceSetStream(occaDevice device, occaStream stream) {
  occa::device device_((occa::device_v*) device);

  device_.setStream(*((occa::stream*) stream));
}

occaStream OCCA_RFUNC occaDeviceWrapStream(occaDevice device, void *handle_) {
  occa::device device_((occa::device_v*) device);

  occa::stream &newStream = *(new occa::stream(device_.wrapStream(handle_)));

  return (occaStream) &newStream;
}

occaStreamTag OCCA_RFUNC occaDeviceTagStream(occaDevice device) {
  occa::device device_((occa::device_v*) device);

  occa::streamTag oldTag = device_.tagStream();
  occaStreamTag newTag;

  ::memcpy(&newTag, &oldTag, sizeof(oldTag));

  return newTag;
}

void OCCA_RFUNC occaDeviceWaitFor(occaDevice device,
                                  occaStreamTag tag) {
  occa::device device_((occa::device_v*) device);

  occa::streamTag tag_;

  ::memcpy(&tag_, &tag, sizeof(tag_));

  device_.waitFor(tag_);
}

double OCCA_RFUNC occaDeviceTimeBetweenTags(occaDevice device,
                                            occaStreamTag startTag, occaStreamTag endTag) {
  occa::device device_((occa::device_v*) device);

  occa::streamTag startTag_, endTag_;

  ::memcpy(&startTag_, &startTag, sizeof(startTag_));
  ::memcpy(&endTag_  , &endTag  , sizeof(endTag_));

  return device_.timeBetween(startTag_, endTag_);
}

void OCCA_RFUNC occaGetStreamFree(occaStream stream) {
  delete (occa::stream*) stream;
}

void OCCA_RFUNC occaStreamFree(occaStream stream) {
  ((occa::stream*) stream)->free();
  delete (occa::stream*) stream;
}

void OCCA_RFUNC occaDeviceFree(occaDevice device) {
  occa::device device_((occa::device_v*) device);

  device_.free();
}
//====================================


//---[ Kernel ]-----------------------
occaDim OCCA_RFUNC occaCreateDim(uintptr_t x, uintptr_t y, uintptr_t z) {
  occaDim ret;

  ret.x = x;
  ret.y = y;
  ret.z = z;

  return ret;
}

const char* OCCA_RFUNC occaKernelMode(occaKernel kernel) {
  occa::kernel kernel_((occa::kernel_v*) kernel);

  return kernel_.mode().c_str();
}

const char* OCCA_RFUNC occaKernelName(occaKernel kernel) {
  occa::kernel kernel_((occa::kernel_v*) kernel);

  return kernel_.name().c_str();
}

occaDevice OCCA_RFUNC occaKernelGetDevice(occaKernel kernel) {
  occa::kernel kernel_((occa::kernel_v*) kernel);
  occa::device device = kernel_.getDevice();

  return (occaDevice) device.getDHandle();
}

uintptr_t OCCA_RFUNC occaKernelMaximumInnerDimSize(occaKernel kernel) {
  occa::kernel kernel_((occa::kernel_v*) kernel);

  return kernel_.maximumInnerDimSize();
}

int OCCA_RFUNC occaKernelPreferredDimSize(occaKernel kernel) {
  occa::kernel kernel_((occa::kernel_v*) kernel);

  return kernel_.preferredDimSize();
}

void OCCA_RFUNC occaKernelSetWorkingDims(occaKernel kernel,
                                         int dims,
                                         occaDim items,
                                         occaDim groups) {

  occa::kernel kernel_((occa::kernel_v*) kernel);

  kernel_.setWorkingDims(dims,
                         occa::dim(items.x, items.y, items.z),
                         occa::dim(groups.x, groups.y, groups.z));
}

void OCCA_RFUNC occaKernelSetAllWorkingDims(occaKernel kernel,
                                            int dims,
                                            uintptr_t itemsX, uintptr_t itemsY, uintptr_t itemsZ,
                                            uintptr_t groupsX, uintptr_t groupsY, uintptr_t groupsZ) {

  occa::kernel kernel_((occa::kernel_v*) kernel);

  kernel_.setWorkingDims(dims,
                         occa::dim(itemsX, itemsY, itemsZ),
                         occa::dim(groupsX, groupsY, groupsZ));
}

occaArgumentList OCCA_RFUNC occaCreateArgumentList() {
  occaArgumentList list = (occaArgumentList) new occaArgumentList_t();
  list->argc = 0;
  return list;
}

void OCCA_RFUNC occaArgumentListClear(occaArgumentList list) {
  occaArgumentList_t &list_ = *list;

  for(int i = 0; i < list_.argc; ++i) {
    if(list_.argv[i]->type != OCCA_TYPE_MEMORY)
      delete list_.argv[i];
  }

  list_.argc = 0;
}

void OCCA_RFUNC occaArgumentListFree(occaArgumentList list) {
  delete list;
}

void OCCA_RFUNC occaArgumentListAddArg(occaArgumentList list,
                                       int argPos,
                                       void *type) {

  occaArgumentList_t &list_ = *list;

  if(list_.argc < (argPos + 1)) {
    OCCA_CHECK(argPos < OCCA_MAX_ARGS,
               "Kernels can only have at most [" << OCCA_MAX_ARGS << "] arguments,"
               << " [" << argPos << "] arguments were set");

    list_.argc = (argPos + 1);
  }

  list_.argv[argPos] = (occaType_t*) type;
}

// Note the _
// [occaKernelRun] is reserved for a variadic macro which is more user-friendly
void OCCA_RFUNC occaKernelRun_(occaKernel kernel,
                               occaArgumentList list) {

  occaArgumentList_t &list_ = *((occaArgumentList_t*) list);
  occaKernelRunN(kernel, list_.argc, list_.argv);
}

void OCCA_RFUNC occaKernelRunN(occaKernel kernel, const int argc, occaType_t **args){
  occa::kernel kernel_((occa::kernel_v*) kernel);
  kernel_.clearArgumentList();

  for(int i = 0; i < argc; ++i){
    occaType_t &arg = *(args[i]);
    void *argPtr = arg.value.data.void_;

    if(arg.type == OCCA_TYPE_MEMORY){
      occa::memory memory_((occa::memory_v*) argPtr);
      kernel_.addArgument(i, occa::kernelArg(memory_));
    }
    else if(arg.type == OCCA_TYPE_PTR){
      occa::memory memory_((void*) argPtr);
      kernel_.addArgument(i, occa::kernelArg(memory_));
    }
    else {
      kernel_.addArgument(i, occa::kernelArg(arg.value));
      delete (occaType_t*) args[i];
    }
  }

  kernel_.runFromArguments();
}

#include "operators/cKernelOperators.cpp"

void OCCA_RFUNC occaKernelFree(occaKernel kernel) {
  occa::kernel kernel_((occa::kernel_v*) kernel);

  kernel_.free();
}

occaKernelInfo OCCA_RFUNC occaCreateKernelInfo() {
  occa::kernelInfo *info = new occa::kernelInfo();

  return (occaKernelInfo) info;
}

void OCCA_RFUNC occaKernelInfoAddDefine(occaKernelInfo info,
                                        const char *macro,
                                        occaType value) {

  occa::kernelInfo &info_ = *((occa::kernelInfo*) info);

  info_.addDefine(macro, occa::typeToStr(value));

  delete value;
}

void OCCA_RFUNC occaKernelInfoAddInclude(occaKernelInfo info,
                                         const char *filename) {

  occa::kernelInfo &info_ = *((occa::kernelInfo*) info);

  info_.addInclude(filename);
}

void OCCA_RFUNC occaKernelInfoFree(occaKernelInfo info) {
  delete (occa::kernelInfo*) info;
}
//====================================


//---[ Helper Functions ]-------------
int OCCA_RFUNC occaSysCall(const char *cmdline,
                           char **output) {
  if(output == NULL)
    return occa::sys::call(cmdline);

  std::string sOutput;
  int ret = occa::sys::call(cmdline, sOutput);

  const size_t chars = sOutput.size();
  *output = (char*) ::malloc(chars + 1);

  ::memcpy(*output, sOutput.c_str(), chars);
  output[chars] = 0;

  return ret;
}
//====================================


//---[ Wrappers ]---------------------
#if OCCA_OPENCL_ENABLED
occaDevice OCCA_RFUNC occaWrapOpenCLDevice(cl_platform_id platformID,
                                           cl_device_id deviceID,
                                           cl_context context) {
  occa::device device = occa::cl::wrapDevice(platformID, deviceID, context);

  return (occaDevice) device.getDHandle();
}
#endif

#if OCCA_CUDA_ENABLED
occaDevice OCCA_RFUNC occaWrapCudaDevice(CUdevice device, CUcontext context) {
  occa::device device_ = occa::cuda::wrapDevice(device, context);

  return (occaDevice) device_.getDHandle();
}
#endif

#if OCCA_HSA_ENABLED
occaDevice OCCA_RFUNC occaWrapHSADevice() {
  occa::device device_ = occa::hsa::wrapDevice();

  return (occaDevice) device_.getDHandle();
}
#endif

occaMemory OCCA_RFUNC occaDeviceWrapMemory(occaDevice device,
                                           void *handle_,
                                           const uintptr_t bytes) {

  occa::device device_((occa::device_v*) device);
  occa::memory memory_ = device_.wrapMemory(handle_, bytes);

  occaMemory memory = (occaMemory) new occaTypePtr_t();
  memory->type()             = OCCA_TYPE_MEMORY;
  memory->value().data.void_ = memory_.getMHandle();
  return memory;
}
//====================================


//---[ Memory ]-----------------------
const char* OCCA_RFUNC occaMemoryMode(occaMemory memory) {
  occa::memory memory_((occa::memory_v*) memory->value().data.void_);

  return memory_.mode().c_str();
}

void* OCCA_RFUNC occaMemoryGetMemoryHandle(occaMemory memory) {
  occa::memory memory_((occa::memory_v*) memory->value().data.void_);

  return memory_.getMemoryHandle();
}

void* OCCA_RFUNC occaMemoryGetMappedPointer(occaMemory memory) {
  occa::memory memory_((occa::memory_v*) memory->value().data.void_);

  return memory_.getMappedPointer();
}

void* OCCA_RFUNC occaMemoryGetTextureHandle(occaMemory memory) {
  occa::memory memory_((occa::memory_v*) memory->value().data.void_);

  return memory_.getTextureHandle();
}

void OCCA_RFUNC occaMemcpy(void *dest, void *src,
                           const uintptr_t bytes) {

  occa::memcpy(dest, src, bytes, occa::autoDetect);
}

void OCCA_RFUNC occaAsyncMemcpy(void *dest, void *src,
                                const uintptr_t bytes) {

  occa::asyncMemcpy(dest, src, bytes, occa::autoDetect);
}

void OCCA_RFUNC occaCopyMemToMem(occaMemory dest, occaMemory src,
                                 const uintptr_t bytes,
                                 const uintptr_t destOffset,
                                 const uintptr_t srcOffset) {

  occa::memory src_((occa::memory_v*) src->value().data.void_);
  occa::memory dest_((occa::memory_v*) dest->value().data.void_);

  memcpy(dest_, src_, bytes, destOffset, srcOffset);
}

void OCCA_RFUNC occaCopyPtrToMem(occaMemory dest, const void *src,
                                 const uintptr_t bytes,
                                 const uintptr_t offset) {

  occa::memory dest_((occa::memory_v*) dest->value().data.void_);

  memcpy(dest_, src, bytes, offset);
}

void OCCA_RFUNC occaCopyMemToPtr(void *dest, occaMemory src,
                                 const uintptr_t bytes,
                                 const uintptr_t offset) {

  occa::memory src_((occa::memory_v*) src->value().data.void_);

  memcpy(dest, src_, bytes, offset);
}

void OCCA_RFUNC occaAsyncCopyMemToMem(occaMemory dest, occaMemory src,
                                      const uintptr_t bytes,
                                      const uintptr_t destOffset,
                                      const uintptr_t srcOffset) {

  occa::memory src_((occa::memory_v*) src->value().data.void_);
  occa::memory dest_((occa::memory_v*) dest->value().data.void_);

  asyncMemcpy(dest_, src_, bytes, destOffset, srcOffset);
}

void OCCA_RFUNC occaAsyncCopyPtrToMem(occaMemory dest, const void * src,
                                      const uintptr_t bytes,
                                      const uintptr_t offset) {

  occa::memory dest_((occa::memory_v*) dest->value().data.void_);

  asyncMemcpy(dest_, src, bytes, offset);
}

void OCCA_RFUNC occaAsyncCopyMemToPtr(void *dest, occaMemory src,
                                      const uintptr_t bytes,
                                      const uintptr_t offset) {

  occa::memory src_((occa::memory_v*) src->value().data.void_);

  asyncMemcpy(dest, src_, bytes, offset);
}

void OCCA_RFUNC occaMemoryFree(occaMemory memory) {
  occa::memory memory_((occa::memory_v*) memory->value().data.void_);
  memory_.free();
  delete memory;
}
//====================================

OCCA_END_EXTERN_C
