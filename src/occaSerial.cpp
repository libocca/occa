#include "occaSerial.hpp"

namespace occa {
  //---[ Kernel ]---------------------
  template <>
  kernel_t<Serial>::kernel_t(){
    data    = NULL;
    dHandle = NULL;

    dims  = 1;
    inner = occa::dim(1,1,1);
    outer = occa::dim(1,1,1);

    nestedKernelCount = 0;
    nestedKernels     = NULL;

    startTime = (void*) new double;
    endTime   = (void*) new double;
  }

  template <>
  kernel_t<Serial>::kernel_t(const kernel_t<Serial> &k){
    data    = k.data;
    dHandle = k.dHandle;

    metaInfo = k.metaInfo;

    dims  = k.dims;
    inner = k.inner;
    outer = k.outer;

    nestedKernelCount = k.nestedKernelCount;
    nestedKernels     = k.nestedKernels;

    for(int i = 0; i < nestedKernelCount; ++i)
      nestedKernels[i] = k.nestedKernels[i];

    startTime = k.startTime;
    endTime   = k.endTime;
  }

  template <>
  kernel_t<Serial>& kernel_t<Serial>::operator = (const kernel_t<Serial> &k){
    data    = k.data;
    dHandle = k.dHandle;

    metaInfo = k.metaInfo;

    dims  = k.dims;
    inner = k.inner;
    outer = k.outer;

    nestedKernelCount = k.nestedKernelCount;
    nestedKernels     = k.nestedKernels;

    for(int i = 0; i < nestedKernelCount; ++i)
      nestedKernels[i] = k.nestedKernels[i];

    *((double*) startTime) = *((double*) k.startTime);
    *((double*) endTime)   = *((double*) k.endTime);

    return *this;
  }

  template <>
  kernel_t<Serial>::~kernel_t(){}

  template <>
  std::string kernel_t<Serial>::getCachedBinaryName(const std::string &filename,
                                                    kernelInfo &info_){

    std::string cachedBinary = getCachedName(filename,
                                             dHandle->getInfoSalt(info_));

#if OCCA_OS == WINDOWS_OS
    // Windows refuses to load dll's that do not end with '.dll'
    cachedBinary = cachedBinary + ".dll";
#endif

    return cachedBinary;
  }

  template <>
  kernel_t<Serial>* kernel_t<Serial>::buildFromSource(const std::string &filename,
                                                      const std::string &functionName,
                                                      const kernelInfo &info_){
    kernelInfo info = info_;

    dHandle->addOccaHeadersToInfo(info);

    std::string cachedBinary = getCachedBinaryName(filename, info);

    if(!haveFile(cachedBinary)){
      waitForFile(cachedBinary);

      if(verboseCompilation_f)
        std::cout << "Found cached binary of [" << filename << "] in [" << cachedBinary << "]\n";

      return buildFromBinary(cachedBinary, functionName);
    }

    struct stat buffer;
    const bool fileExists = (stat(cachedBinary.c_str(), &buffer) == 0);

    if(fileExists){
      releaseFile(cachedBinary);

      if(verboseCompilation_f)
        std::cout << "Found cached binary of [" << filename << "] in [" << cachedBinary << "]\n";

      return buildFromBinary(cachedBinary, functionName);
    }

    data = new SerialKernelData_t;

    //---[ Create Intermediate ]--------
    const std::string iCachedBinary = getMidCachedBinaryName(cachedBinary, "i");

    {
      struct stat buffer;
      bool fileExists = (stat(iCachedBinary.c_str(), &buffer) == 0);

      if(!fileExists){
        std::ofstream fs;
        fs.open(iCachedBinary.c_str());

        fs << info.occaKeywords << info.header << readFile(filename);

        fs.close();
      }
    }
    //==================================

    const std::string occaDir = getOCCADir();

    std::stringstream command;

    if(dHandle->compilerEnvScript.size())
      command << dHandle->compilerEnvScript << " && ";

    command << dHandle->compiler
#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
            << " -x c++ -w -fPIC -shared"
#else
            << " /TP /LD /D MC_CL_EXE"
#endif
            << ' '    << dHandle->compilerFlags
            << ' '    << info.flags
            << " -I"  << occaDir << "/include"
            << " -L"  << occaDir << "/lib -locca"
            << ' '    << iCachedBinary
#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
            << " -o " << cachedBinary
#else
            << " /link /OUT:" << cachedBinary
#endif
            << std::endl;

    const std::string &sCommand = command.str();

    if(verboseCompilation_f)
      std::cout << "Compiling [" << functionName << "]\n" << sCommand << "\n";

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
    const int compileError = system(sCommand.c_str());
#else
    const int compileError = system(("\"" +  sCommand + "\"").c_str());
#endif

    if(compileError){
      releaseFile(cachedBinary);
      OCCA_CHECK(false, "Compilation error");
    }

    OCCA_EXTRACT_DATA(Serial, Kernel);

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
    data_.dlHandle = dlopen(cachedBinary.c_str(), RTLD_NOW);

    if(data_.dlHandle == NULL){
      releaseFile(cachedBinary);
      OCCA_CHECK(false, "Error loading binary using dlopen");
    }
#else
    data_.dlHandle = LoadLibraryA(cachedBinary.c_str());

    if(data_.dlHandle == NULL){
      releaseFile(cachedBinary);

      OCCA_CHECK(false,
                 "Error loading dll [" << filename << "] (WIN32 error: " << GetLastError() << ")");
    }
#endif

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
    data_.handle = dlsym(data_.dlHandle, functionName.c_str());

    char *dlError;
    if ((dlError = dlerror()) != NULL)  {
      releaseFile(cachedBinary);
      OCCA_CHECK(false, "Error loading symbol from binary with dlsym (DL Error: " << dlError << ")");
    }
#else
    data_.handle = GetProcAddress((HMODULE) (data_.dlHandle), functionName.c_str());

    if(data_.handle == NULL) {
      releaseFile(cachedBinary);
      OCCA_CHECK(false, "Error loading symbol from binary with GetProcAddress");
    }
#endif

    releaseFile(cachedBinary);

    return this;
  }

  template <>
  kernel_t<Serial>* kernel_t<Serial>::buildFromBinary(const std::string &filename,
                                                      const std::string &functionName){
    data = new SerialKernelData_t;

    OCCA_EXTRACT_DATA(Serial, Kernel);

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
    data_.dlHandle = dlopen(filename.c_str(), RTLD_NOW);

    OCCA_CHECK(data_.dlHandle != NULL,
               "Error loading binary using dlopen");
#else
    data_.dlHandle = LoadLibraryA(filename.c_str());

    OCCA_CHECK(false,
               "Error loading dll [" << filename << "] (WIN32 error: " << GetLastError() << ")");
#endif

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
    data_.handle = dlsym(data_.dlHandle, functionName.c_str());

    char *dlError;
    if ((dlError = dlerror()) != NULL)  {
      OCCA_CHECK(false, "Error loading symbol from binary with dlsym (DL Error: " << dlError << ")");
    }
#else
    data_.handle = GetProcAddress((HMODULE) (data_.dlHandle), functionName.c_str());

    if(data_.handle == NULL) {
      OCCA_CHECK(false, "Error loading symbol from binary with GetProcAddress");
    }
#endif

    return this;
  }

  template <>
  kernel_t<Serial>* kernel_t<Serial>::loadFromLibrary(const char *cache,
                                                      const std::string &functionName){
    return buildFromBinary(cache, functionName);
  }

  // [-] Missing
  template <>
  int kernel_t<Serial>::preferredDimSize(){
    preferredDimSize_ = OCCA_SIMD_WIDTH;
    return OCCA_SIMD_WIDTH;
  }

#include "operators/occaSerialKernelOperators.cpp"

  template <>
  double kernel_t<Serial>::timeTaken(){
    const double &start = *((double*) startTime);
    const double &end   = *((double*) endTime);

    return 1.0e3*(end - start);
  }

  template <>
  double kernel_t<Serial>::timeTakenBetween(void *start, void *end){
    const double &start_ = *((double*) start);
    const double &end_   = *((double*) end);

    return 1.0e3*(end_ - start_);
  }

  template <>
  void kernel_t<Serial>::free(){
    OCCA_EXTRACT_DATA(Serial, Kernel);

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
    dlclose(data_.dlHandle);
#else
    FreeLibrary((HMODULE) (data_.dlHandle));
#endif
  }
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<Serial>::memory_t(){
    handle    = NULL;
    mappedPtr = NULL;
    uvaPtr    = NULL;

    dHandle = NULL;
    size    = 0;

    isTexture = false;
    textureInfo.arg = NULL;
    textureInfo.dim = 1;
    textureInfo.w = textureInfo.h = textureInfo.d = 0;

    uva_inDevice = false;
    uva_isDirty  = false;

    isManaged  = false;
    isMapped   = false;
    isAWrapper = false;
  }

  template <>
  memory_t<Serial>::memory_t(const memory_t<Serial> &m){
    *this = m;
  }

  template <>
  memory_t<Serial>& memory_t<Serial>::operator = (const memory_t<Serial> &m){
    handle    = m.handle;
    mappedPtr = m.mappedPtr;
    uvaPtr    = m.uvaPtr;

    dHandle = m.dHandle;
    size    = m.size;

    isTexture = m.isTexture;
    textureInfo.arg  = m.textureInfo.arg;
    textureInfo.dim  = m.textureInfo.dim;

    textureInfo.w = m.textureInfo.w;
    textureInfo.h = m.textureInfo.h;
    textureInfo.d = m.textureInfo.d;

    if(isTexture)
      handle = &textureInfo;

    uva_inDevice = m.uva_inDevice;
    uva_isDirty  = m.uva_isDirty;

    isManaged  = m.isManaged;
    isMapped   = m.isMapped;
    isAWrapper = m.isAWrapper;

    return *this;
  }

  template <>
  memory_t<Serial>::~memory_t(){}

  template <>
  void* memory_t<Serial>::getMemoryHandle(){
    return handle;
  }

  template <>
  void* memory_t<Serial>::getTextureHandle(){
    return textureInfo.arg;
  }

  template <>
  void memory_t<Serial>::copyFrom(const void *src,
                                  const uintptr_t bytes,
                                  const uintptr_t offset){
    dHandle->finish();

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

    void *destPtr      = ((char*) (isTexture ? textureInfo.arg : handle)) + offset;
    const void *srcPtr = src;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<Serial>::copyFrom(const memory_v *src,
                                  const uintptr_t bytes,
                                  const uintptr_t destOffset,
                                  const uintptr_t srcOffset){
    dHandle->finish();

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + destOffset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << destOffset << " , " << (destOffset + bytes_) << " ]");

    OCCA_CHECK((bytes_ + srcOffset) <= src->size,
               "Source has size [" << src->size << "],"
               << "trying to access [ " << srcOffset << " , " << (srcOffset + bytes_) << " ]");

    void *destPtr      = ((char*) (isTexture      ? textureInfo.arg      : handle))      + destOffset;
    const void *srcPtr = ((char*) (src->isTexture ? src->textureInfo.arg : src->handle)) + srcOffset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<Serial>::copyTo(void *dest,
                                const uintptr_t bytes,
                                const uintptr_t offset){
    dHandle->finish();

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

    void *destPtr      = dest;
    const void *srcPtr = ((char*) (isTexture ? textureInfo.arg : handle)) + offset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<Serial>::copyTo(memory_v *dest,
                                const uintptr_t bytes,
                                const uintptr_t destOffset,
                                const uintptr_t srcOffset){
    dHandle->finish();

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + srcOffset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << srcOffset << " , " << (srcOffset + bytes_) << " ]");

    OCCA_CHECK((bytes_ + destOffset) <= dest->size,
               "Destination has size [" << dest->size << "],"
               << "trying to access [ " << destOffset << " , " << (destOffset + bytes_) << " ]");

    void *destPtr      = ((char*) (dest->isTexture ? dest->textureInfo.arg : dest->handle)) + destOffset;
    const void *srcPtr = ((char*) (isTexture       ? textureInfo.arg       : handle))       + srcOffset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<Serial>::asyncCopyFrom(const void *src,
                                       const uintptr_t bytes,
                                       const uintptr_t offset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

    void *destPtr      = ((char*) (isTexture ? textureInfo.arg : handle)) + offset;
    const void *srcPtr = src;


    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<Serial>::asyncCopyFrom(const memory_v *src,
                                       const uintptr_t bytes,
                                       const uintptr_t destOffset,
                                       const uintptr_t srcOffset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + destOffset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << destOffset << " , " << (destOffset + bytes_) << " ]");

    OCCA_CHECK((bytes_ + srcOffset) <= src->size,
               "Source has size [" << src->size << "],"
               << "trying to access [ " << srcOffset << " , " << (srcOffset + bytes_) << " ]");

    void *destPtr      = ((char*) (isTexture      ? textureInfo.arg      : handle))         + destOffset;
    const void *srcPtr = ((char*) (src->isTexture ? src->textureInfo.arg : src->handle)) + srcOffset;;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<Serial>::asyncCopyTo(void *dest,
                                     const uintptr_t bytes,
                                     const uintptr_t offset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

    void *destPtr      = dest;
    const void *srcPtr = ((char*) (isTexture ? textureInfo.arg : handle)) + offset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<Serial>::asyncCopyTo(memory_v *dest,
                                     const uintptr_t bytes,
                                     const uintptr_t destOffset,
                                     const uintptr_t srcOffset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + srcOffset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << srcOffset << " , " << (srcOffset + bytes_) << " ]");

    OCCA_CHECK((bytes_ + destOffset) <= dest->size,
               "Destination has size [" << dest->size << "],"
               << "trying to access [ " << destOffset << " , " << (destOffset + bytes_) << " ]");

    void *destPtr      = ((char*) (dest->isTexture ? dest->textureInfo.arg : dest->handle)) + destOffset;
    const void *srcPtr = ((char*) (isTexture       ? textureInfo.arg       : handle))       + srcOffset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<Serial>::mappedFree(){
    ::free(handle);
    handle    = NULL;
    mappedPtr = NULL;

    size = 0;
  }

  template <>
  void memory_t<Serial>::free(){
    if(isTexture){
      ::free(textureInfo.arg);
      textureInfo.arg = NULL;
    }
    else{
      ::free(handle);
      handle = NULL;
    }

    size = 0;
  }
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<Serial>::device_t(){
    data = NULL;

    uvaEnabled_ = false;

    bytesAllocated = 0;

    getEnvironmentVariables();
  }

  template <>
  device_t<Serial>::device_t(const device_t<Serial> &d){
    *this = d;
  }

  template <>
  device_t<Serial>& device_t<Serial>::operator = (const device_t<Serial> &d){
    modelID_ = d.modelID_;
    id_      = d.id_;

    data = d.data;

    uvaEnabled_    = d.uvaEnabled_;
    uvaMap         = d.uvaMap;
    uvaDirtyMemory = d.uvaDirtyMemory;

    compiler      = d.compiler;
    compilerFlags = d.compilerFlags;

    bytesAllocated = d.bytesAllocated;

    return *this;
  }

  template <>
  void device_t<Serial>::setup(argInfoMap &aim){}

  template <>
  void device_t<Serial>::addOccaHeadersToInfo(kernelInfo &info_){
    info_.addDefine("OCCA_USING_CPU", 1);

    info_.addOCCAKeywords(occaSerialDefines);
  }

  template <>
  std::string device_t<Serial>::getInfoSalt(const kernelInfo &info_){
    std::stringstream salt;

    salt << "Serial"
         << info_.salt()
         << parser::version
         << compilerEnvScript
         << compiler
         << compilerFlags;

    return salt.str();
  }

  template <>
  deviceIdentifier device_t<Serial>::getIdentifier() const {
    deviceIdentifier dID;

    dID.mode_ = Serial;

    const bool debugEnabled = (compilerFlags.find("-g") != std::string::npos);

    dID.flagMap["compiler"]     = compiler;
    dID.flagMap["debugEnabled"] = (debugEnabled ? "true" : "false");

    for(int i = 0; i <= 3; ++i){
      std::string flag = "-O";
      flag += '0' + i;

      if(compilerFlags.find(flag) != std::string::npos){
        dID.flagMap["optimization"] = '0' + i;
        break;
      }

      if(i == 3)
        dID.flagMap["optimization"] = "None";
    }

    return dID;
  }

  template <>
  void device_t<Serial>::getEnvironmentVariables(){
    char *c_compiler = getenv("OCCA_OPENMP_COMPILER");

    if(c_compiler != NULL)
      compiler = std::string(c_compiler);
    else
#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
      compiler = "g++";
#else
      compiler = "cl.exe";
#endif

    char *c_compilerFlags = getenv("OCCA_OPENMP_COMPILER_FLAGS");

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
    if(c_compilerFlags != NULL)
      compilerFlags = std::string(c_compilerFlags);
    else{
#  if OCCA_DEBUG_ENABLED
      compilerFlags = "-g";
#  else
      compilerFlags = "";
#  endif
    }
#else
#  if OCCA_DEBUG_ENABLED
    compilerFlags = " /Od ";
#  endif
    std::string byteness;

    if(sizeof(void*) == 4)
      byteness = "x86 ";
    else if(sizeof(void*) == 8)
      byteness = "amd64";
    else
      OCCA_CHECK(false, "sizeof(void*) is not equal to 4 or 8");

    char* visual_studio_tools = getenv("VS100COMNTOOLS");
    if(visual_studio_tools != NULL){
      setCompilerEnvScript("\"" + std::string(visual_studio_tools) + "\\..\\..\\VC\\vcvarsall.bat\" " + byteness);
    }
    else{
      std::cout << "WARNING: VS100COMNTOOLS environment variable not found -> compiler environment (vcvarsall.bat) maybe not correctly setup." << std::endl;
    }
#endif
  }

  template <>
  void device_t<Serial>::appendAvailableDevices(std::vector<device> &dList){
    device d;
    d.setup("Serial");

    dList.push_back(d);
  }

  template <>
  void device_t<Serial>::setCompiler(const std::string &compiler_){
    compiler = compiler_;
  }

  template <>
  void device_t<Serial>::setCompilerEnvScript(const std::string &compilerEnvScript_){
    compilerEnvScript = compilerEnvScript_;
  }

  template <>
  void device_t<Serial>::setCompilerFlags(const std::string &compilerFlags_){
    compilerFlags = compilerFlags_;
  }

  template <>
  std::string& device_t<Serial>::getCompiler(){
    return compiler;
  }

  template <>
  std::string& device_t<Serial>::getCompilerEnvScript(){
    return compilerEnvScript;
  }

  template <>
  std::string& device_t<Serial>::getCompilerFlags(){
    return compilerFlags;
  }

  template <>
  void device_t<Serial>::flush(){}

  template <>
  void device_t<Serial>::finish(){}

  template <>
  bool device_t<Serial>::fakesUva(){
    return false;
  }

  template <>
  void device_t<Serial>::waitFor(tag tag_){}

  template <>
  stream device_t<Serial>::createStream(){
    return NULL;
  }

  template <>
  void device_t<Serial>::freeStream(stream s){}

  template <>
  stream device_t<Serial>::wrapStream(void *handle_){
    return NULL;
  }

  template <>
  tag device_t<Serial>::tagStream(){
    tag ret;

    ret.tagTime = currentTime();

    return ret;
  }

  template <>
  double device_t<Serial>::timeBetween(const tag &startTag, const tag &endTag){
    return (endTag.tagTime - startTag.tagTime);
  }

  template <>
  kernel_v* device_t<Serial>::buildKernelFromSource(const std::string &filename,
                                                    const std::string &functionName,
                                                    const kernelInfo &info_){
    kernel_v *k = new kernel_t<Serial>;
    k->dHandle = this;

    k->buildFromSource(filename, functionName, info_);

    return k;
  }

  template <>
  kernel_v* device_t<Serial>::buildKernelFromBinary(const std::string &filename,
                                                    const std::string &functionName){
    kernel_v *k = new kernel_t<Serial>;
    k->dHandle = this;
    k->buildFromBinary(filename, functionName);
    return k;
  }

  template <>
  void device_t<Serial>::cacheKernelInLibrary(const std::string &filename,
                                              const std::string &functionName,
                                              const kernelInfo &info_){
    //---[ Creating shared library ]----
    kernel tmpK = occa::device(this).buildKernelFromSource(filename, functionName, info_);
    tmpK.free();

    kernelInfo info = info_;

    addOccaHeadersToInfo(info);

    std::string cachedBinary = getCachedName(filename, getInfoSalt(info));

#if OCCA_OS == WINDOWS_OS
    // Windows refuses to load dll's that do not end with '.dll'
    cachedBinary = cachedBinary + ".dll";
#endif
    //==================================

    library::infoID_t infoID;

    infoID.modelID    = modelID_;
    infoID.kernelName = functionName;

    library::infoHeader_t &header = library::headerMap[infoID];

    header.fileID = -1;
    header.mode   = Serial;

    const std::string flatDevID = getIdentifier().flattenFlagMap();

    header.flagsOffset = library::addToScratchPad(flatDevID);
    header.flagsBytes  = flatDevID.size();

    header.contentOffset = library::addToScratchPad(cachedBinary);
    header.contentBytes  = cachedBinary.size();

    header.kernelNameOffset = library::addToScratchPad(functionName);
    header.kernelNameBytes  = functionName.size();
  }

  template <>
  kernel_v* device_t<Serial>::loadKernelFromLibrary(const char *cache,
                                                    const std::string &functionName){
    kernel_v *k = new kernel_t<Serial>;
    k->dHandle = this;
    k->loadFromLibrary(cache, functionName);
    return k;
  }

  template <>
  memory_v* device_t<Serial>::wrapMemory(void *handle_,
                                         const uintptr_t bytes){
    memory_v *mem = new memory_t<Serial>;

    mem->dHandle = this;
    mem->size    = bytes;
    mem->handle  = handle_;

    mem->isAWrapper = true;

    return mem;
  }

  template <>
  memory_v* device_t<Serial>::wrapTexture(void *handle_,
                                          const int dim, const occa::dim &dims,
                                          occa::formatType type, const int permissions){
    memory_v *mem = new memory_t<Serial>;

    mem->dHandle = this;
    mem->size    = ((dim == 1) ? dims.x : (dims.x * dims.y)) * type.bytes();

    mem->isTexture = true;
    mem->textureInfo.dim  = dim;

    mem->textureInfo.w = dims.x;
    mem->textureInfo.h = dims.y;
    mem->textureInfo.d = dims.z;

    mem->textureInfo.arg = handle_;

    mem->handle = &(mem->textureInfo);

    mem->isAWrapper = true;

    return mem;
  }

  template <>
  memory_v* device_t<Serial>::malloc(const uintptr_t bytes,
                                     void *src){
    memory_v *mem = new memory_t<Serial>;

    mem->dHandle = this;
    mem->size    = bytes;

#if   (OCCA_OS == LINUX_OS)
    posix_memalign(&(mem->handle), OCCA_MEM_ALIGN, bytes);
#elif (OCCA_OS == OSX_OS)
    mem->handle = ::malloc(bytes);
#else
    mem->handle = ::malloc(bytes);
#endif

    if(src != NULL)
      ::memcpy(mem->handle, src, bytes);

    return mem;
  }

  template <>
  memory_v* device_t<Serial>::textureAlloc(const int dim, const occa::dim &dims,
                                           void *src,
                                           occa::formatType type, const int permissions){
    memory_v *mem = new memory_t<Serial>;

    mem->dHandle = this;
    mem->size    = ((dim == 1) ? dims.x : (dims.x * dims.y)) * type.bytes();

    mem->isTexture = true;
    mem->textureInfo.dim  = dim;

    mem->textureInfo.w = dims.x;
    mem->textureInfo.h = dims.y;
    mem->textureInfo.d = dims.z;

#if   OCCA_OS == LINUX_OS
    posix_memalign(&(mem->textureInfo.arg), OCCA_MEM_ALIGN, mem->size);
#elif OCCA_OS == OSX_OS
    mem->textureInfo.arg = ::malloc(mem->size);
#else
    mem->textureInfo.arg = ::malloc(mem->size);
#endif

    ::memcpy(mem->textureInfo.arg, src, mem->size);

    mem->handle = &(mem->textureInfo);

    return mem;
  }

  template <>
  memory_v* device_t<Serial>::mappedAlloc(const uintptr_t bytes,
                                          void *src){
    memory_v *mem = malloc(bytes, src);

    mem->mappedPtr = mem->handle;

    return mem;
  }

  template <>
  void device_t<Serial>::free(){}

  template <>
  int device_t<Serial>::simdWidth(){
    simdWidth_ = OCCA_SIMD_WIDTH;
    return OCCA_SIMD_WIDTH;
  }
  //==================================
};
