#include "occaSerial.hpp"
#include "occaPthreads.hpp"

namespace occa {
  //---[ Kernel ]---------------------
  template <>
  kernel_t<Pthreads>::kernel_t(){
    data    = NULL;
    dHandle = NULL;

    dims  = 1;
    inner = occa::dim(1,1,1);
    outer = occa::dim(1,1,1);

    nestedKernelCount = 0;
  }

  template <>
  kernel_t<Pthreads>::kernel_t(const kernel_t<Pthreads> &k){
    data    = k.data;
    dHandle = k.dHandle;

    metaInfo = k.metaInfo;

    dims  = k.dims;
    inner = k.inner;
    outer = k.outer;

    nestedKernelCount = k.nestedKernelCount;

    if(nestedKernelCount){
      nestedKernels = new kernel[nestedKernelCount];

      for(int i = 0; i < nestedKernelCount; ++i)
        nestedKernels[i] = k.nestedKernels[i];
    }
  }

  template <>
  kernel_t<Pthreads>& kernel_t<Pthreads>::operator = (const kernel_t<Pthreads> &k){
    data    = k.data;
    dHandle = k.dHandle;

    metaInfo = k.metaInfo;

    dims  = k.dims;
    inner = k.inner;
    outer = k.outer;

    nestedKernelCount = k.nestedKernelCount;

    if(nestedKernelCount){
      nestedKernels = new kernel[nestedKernelCount];

      for(int i = 0; i < nestedKernelCount; ++i)
        nestedKernels[i] = k.nestedKernels[i];
    }

    return *this;
  }

  template <>
  kernel_t<Pthreads>::~kernel_t(){}

  template <>
  std::string kernel_t<Pthreads>::getCachedBinaryName(const std::string &filename,
                                                      kernelInfo &info_){

    std::string cachedBinary = getCachedName(filename,
                                             dHandle->getInfoSalt(info_));

#if (OCCA_OS & WINDOWS_OS)
    // Windows requires .dll extension
    cachedBinary = cachedBinary + ".dll";
#endif

    return cachedBinary;
  }

  template <>
  kernel_t<Pthreads>* kernel_t<Pthreads>::buildFromSource(const std::string &filename,
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
    bool fileExists = (stat(cachedBinary.c_str(), &buffer) == 0);

    if(fileExists){
      releaseFile(cachedBinary);

      if(verboseCompilation_f)
        std::cout << "Found cached binary of [" << filename << "] in [" << cachedBinary << "]\n";

      return buildFromBinary(cachedBinary, functionName);
    }

    data = new PthreadsKernelData_t;

    std::string iCachedBinary = createIntermediateSource(filename,
                                                         cachedBinary,
                                                         info);

    const std::string occaDir = getOCCADir();

    std::stringstream command;

    if(dHandle->compilerEnvScript.size())
      command << dHandle->compilerEnvScript << " && ";

#if (OCCA_OS & (LINUX_OS | OSX_OS))
    command << dHandle->compiler
            << ' '    << dHandle->compilerFlags
            << ' '    << info.flags
            << ' '    << iCachedBinary
            << " -o " << cachedBinary
            << " -I"  << occaDir << "/include"
            << " -L"  << occaDir << "/lib -locca"
            << std::endl;
#else
#  if (OCCA_DEBUG_ENABLED)
    std::string occaLib = occaDir + "\\lib\\libocca_d.lib ";
#  else
    std::string occaLib = occaDir + "\\lib\\libocca.lib ";
#  endif
    std::string ptLib = occaDir + "\\lib\\pthreadVC2.lib ";

    command << dHandle->compiler
            << " /D MC_CL_EXE"
            << ' '    << dHandle->compilerFlags
            << ' '    << info.flags
            << " /I"  << occaDir << "\\include"
            << ' '    << iCachedBinary
            << " /link " << occaLib << ptLib << " /OUT:" << cachedBinary
            << std::endl;
#endif

    const std::string &sCommand = command.str();

    if(verboseCompilation_f)
      std::cout << "Compiling [" << functionName << "]\n" << sCommand << "\n";

#if (OCCA_OS & (LINUX_OS | OSX_OS))
    const int compileError = system(sCommand.c_str());
#else
    const int compileError = system(("\"" +  sCommand + "\"").c_str());
#endif

    if(compileError){
      releaseFile(cachedBinary);
      OCCA_CHECK(false, "Compilation error");
    }

    OCCA_EXTRACT_DATA(Pthreads, Kernel);

    data_.dlHandle = cpu::dlopen(cachedBinary, true);
    data_.handle   = cpu::dlsym(data_.dlHandle, cachedBinary, functionName, true);

    PthreadsDeviceData_t &dData = *((PthreadsDeviceData_t*) ((device_t<Pthreads>*) dHandle)->data);

    data_.pThreadCount = dData.pThreadCount;

    data_.pendingJobs = &(dData.pendingJobs);

    for(int p = 0; p < 50; ++p){
      data_.kernelLaunch[p] = &(dData.kernelLaunch[p]);
      data_.kernelArgs[p]   = &(dData.kernelArgs[p]);
    }

    data_.pendingJobsMutex = &(dData.pendingJobsMutex);
    data_.kernelMutex      = &(dData.kernelMutex);

    releaseFile(cachedBinary);

    return this;
  }

  template <>
  kernel_t<Pthreads>* kernel_t<Pthreads>::buildFromBinary(const std::string &filename,
                                                          const std::string &functionName){
    data = new PthreadsKernelData_t;

    OCCA_EXTRACT_DATA(Pthreads, Kernel);

    data_.dlHandle = cpu::dlopen(filename, false);
    data_.handle   = cpu::dlsym(data_.dlHandle, filename, functionName, false);

    PthreadsDeviceData_t &dData = *((PthreadsDeviceData_t*) ((device_t<Pthreads>*) dHandle)->data);

    data_.pThreadCount = dData.pThreadCount;

    data_.pendingJobs = &(dData.pendingJobs);

    for(int p = 0; p < 50; ++p){
      data_.kernelLaunch[p] = &(dData.kernelLaunch[p]);
      data_.kernelArgs[p]   = &(dData.kernelArgs[p]);
    }

    data_.pendingJobsMutex = &(dData.pendingJobsMutex);
    data_.kernelMutex      = &(dData.kernelMutex);

    return this;
  }

  template <>
  kernel_t<Pthreads>* kernel_t<Pthreads>::loadFromLibrary(const char *cache,
                                                          const std::string &functionName){
    return buildFromBinary(cache, functionName);
  }

  template <>
  uintptr_t kernel_t<Pthreads>::maximumInnerDimSize(){
    return ((uintptr_t) -1);
  }

  // [-] Missing
  template <>
  int kernel_t<Pthreads>::preferredDimSize(){
    preferredDimSize_ = OCCA_SIMD_WIDTH;
    return OCCA_SIMD_WIDTH;
  }

  template <>
  void kernel_t<Pthreads>::free(){
    // [-] Fix later
    OCCA_EXTRACT_DATA(Pthreads, Kernel);

#if (OCCA_OS & (LINUX_OS | OSX_OS))
    dlclose(data_.dlHandle);
#else
    FreeLibrary((HMODULE) (data_.dlHandle));
#endif
  }
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<Pthreads>::memory_t(){
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
  memory_t<Pthreads>::memory_t(const memory_t<Pthreads> &m){
    *this = m;
  }

  template <>
  memory_t<Pthreads>& memory_t<Pthreads>::operator = (const memory_t<Pthreads> &m){
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
  memory_t<Pthreads>::~memory_t(){}

  template <>
  void* memory_t<Pthreads>::getMemoryHandle(){
    return handle;
  }

  template <>
  void* memory_t<Pthreads>::getTextureHandle(){
    return textureInfo.arg;
  }

  template <>
  void memory_t<Pthreads>::copyFrom(const void *src,
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
  void memory_t<Pthreads>::copyFrom(const memory_v *src,
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
  void memory_t<Pthreads>::copyTo(void *dest,
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
  void memory_t<Pthreads>::copyTo(memory_v *dest,
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
    const void *srcPtr = ((char*) (isTexture ? textureInfo.arg : handle))       + srcOffset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<Pthreads>::asyncCopyFrom(const void *src,
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
  void memory_t<Pthreads>::asyncCopyFrom(const memory_v *src,
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

    void *destPtr      = ((char*) (isTexture      ? textureInfo.arg      : handle))      + destOffset;
    const void *srcPtr = ((char*) (src->isTexture ? src->textureInfo.arg : src->handle)) + srcOffset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<Pthreads>::asyncCopyTo(void *dest,
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
  void memory_t<Pthreads>::asyncCopyTo(memory_v *dest,
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
    const void *srcPtr = ((char*) (isTexture ? textureInfo.arg : handle))       + srcOffset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<Pthreads>::mappedFree(){
    cpu::free(handle);
    handle    = NULL;
    mappedPtr = NULL;

    size = 0;
  }

  template <>
  void memory_t<Pthreads>::free(){
    if(isTexture){
      cpu::free(textureInfo.arg);
      textureInfo.arg = NULL;
    }
    else{
      cpu::free(handle);
      handle = NULL;
    }

    size = 0;
  }
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<Pthreads>::device_t(){
    data = NULL;

    uvaEnabled_ = false;

    bytesAllocated = 0;

    getEnvironmentVariables();

    cpu::addSharedBinaryFlagsTo(compiler, compilerFlags);
  }

  template <>
  device_t<Pthreads>::device_t(const device_t<Pthreads> &d){
    data           = d.data;
    bytesAllocated = d.bytesAllocated;

    compiler      = d.compiler;
    compilerFlags = d.compilerFlags;
  }

  template <>
  device_t<Pthreads>& device_t<Pthreads>::operator = (const device_t<Pthreads> &d){
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
  void device_t<Pthreads>::setup(argInfoMap &aim){
    data = new PthreadsDeviceData_t;

    OCCA_EXTRACT_DATA(Pthreads, Device);

    data_.vendor = cpu::compilerVendor(compiler);

    cpu::addSharedBinaryFlagsTo(data_.vendor, compilerFlags);

    data_.pendingJobs = 0;

    data_.coreCount = cpu::getCoreCount();

    std::vector<int> pinnedCores;

    if(!aim.has("threadCount"))
      data_.pThreadCount = 1;
    else
      data_.pThreadCount = aim.iGet("threadCount");

    if(!aim.has("schedule") ||
       (aim.get("schedule") == "compact")){

      data_.schedule = occa::compact;
    }
    else{
      data_.schedule = occa::scatter;
    }

    if(aim.has("pinnedCores")){
      aim.iGets("pinnedCores", pinnedCores);

      if(pinnedCores.size() != (size_t) data_.pThreadCount){
        std::cout << "[Pthreads]: Mismatch between thread count and pinned cores\n"
                  << "            Defaulting to ["
                  << ((data_.schedule == occa::compact) ?
                      "compact" : "scatter")
                  << "] scheduling\n"
                  << "  Thread Count: " << data_.pThreadCount << '\n'
                  << "  Pinned Cores: [";

        if(pinnedCores.size()){
          std::cout << pinnedCores[0];

          for(size_t i = 1; i < pinnedCores.size(); ++i)
            std::cout << ", " << pinnedCores[i];
        }

        std::cout << "]\n";

        pinnedCores.clear();
      }
      else{
        for(size_t i = 0; i < pinnedCores.size(); ++i)
          if(pinnedCores[i] < 0){
            const int newPC = (((pinnedCores[i] % data_.coreCount)
                                + pinnedCores[i]) % data_.coreCount);

            std::cout << "Trying to pin thread on core ["
                      << pinnedCores[i] << "], changing it to ["
                      << newPC << "]\n";

            pinnedCores[i] = newPC;
          }
          else if(data_.coreCount <= pinnedCores[i]){
            const int newPC = (pinnedCores[i] % data_.coreCount);

            std::cout << "Trying to pin thread on core ["
                      << pinnedCores[i] << "], changing it to ["
                      << newPC << "]\n";

            pinnedCores[i] = newPC;
          }

        data_.schedule = occa::manual;
      }
    }

    int error = pthread_mutex_init(&(data_.pendingJobsMutex), NULL);
    OCCA_CHECK(error == 0, "Error initializing mutex");

    error = pthread_mutex_init(&(data_.kernelMutex), NULL);
    OCCA_CHECK(error == 0, "Error initializing mutex");

    for(int p = 0; p < data_.pThreadCount; ++p){
      PthreadWorkerData_t *args = new PthreadWorkerData_t;

      args->rank  = p;
      args->count = data_.pThreadCount;

      // [-] Need to know number of sockets
      if(data_.schedule & occa::compact)
        args->pinnedCore = p % data_.coreCount;
      else if(data_.schedule & occa::scatter)
        args->pinnedCore = p % data_.coreCount;
      else // manual
        args->pinnedCore = pinnedCores[p];

      args->pendingJobs = &(data_.pendingJobs);

      args->pendingJobsMutex = &(data_.pendingJobsMutex);
      args->kernelMutex      = &(data_.kernelMutex);

      args->kernelLaunch = &(data_.kernelLaunch[p]);
      args->kernelArgs   = &(data_.kernelArgs[p]);

      pthread_create(&data_.tid[p], NULL, pthreadLimbo, args);
    }
  }

  template <>
  void device_t<Pthreads>::addOccaHeadersToInfo(kernelInfo &info_){
    info_.addOCCAKeywords(occaPthreadsDefines);
  }

  template <>
  std::string device_t<Pthreads>::getInfoSalt(const kernelInfo &info_){
    std::stringstream salt;

    salt << "Pthreads"
         << info_.salt()
         << parserVersion
         << compilerEnvScript
         << compiler
         << compilerFlags;

    return salt.str();
  }

  template <>
  deviceIdentifier device_t<Pthreads>::getIdentifier() const {
    deviceIdentifier dID;

    dID.mode_ = Pthreads;

#if (OCCA_OS & (LINUX_OS | OSX_OS))
    const bool debugEnabled = (compilerFlags.find("-g") != std::string::npos);
#else
    const bool debugEnabled = (compilerFlags.find("/Od") != std::string::npos);
#endif

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
  void device_t<Pthreads>::getEnvironmentVariables(){
    char *c_compiler = getenv("OCCA_CXX");

    if(c_compiler != NULL){
      compiler = std::string(c_compiler);
    }
    else{
      c_compiler = getenv("CXX");

      if(c_compiler != NULL){
        compiler = std::string(c_compiler);
      }
      else{
#if (OCCA_OS & (LINUX_OS | OSX_OS))
        compiler = "g++";
#else
        compiler = "cl.exe";
#endif
      }
    }

    char *c_compilerFlags = getenv("OCCA_CXXFLAGS");

#if (OCCA_OS & (LINUX_OS | OSX_OS))
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
    compilerFlags = " /Od";
#  else
    compilerFlags = " /O2";
#  endif

    std::string byteness;

    if(sizeof(void*) == 4)
      byteness = "x86 ";
    else if(sizeof(void*) == 8)
      byteness = "amd64";
    else
      OCCA_CHECK(false, "sizeof(void*) is not equal to 4 or 8");

#  if      (OCCA_VS_VERSION == 1800)
    char *visualStudioTools = getenv("VS120COMNTOOLS");   // MSVC++ 12.0 - Visual Studio 2013
#  elif    (OCCA_VS_VERSION == 1700)
    char *visualStudioTools = getenv("VS110COMNTOOLS");   // MSVC++ 11.0 - Visual Studio 2012
#  else // (OCCA_VS_VERSION == 1600)
    char *visualStudioTools = getenv("VS100COMNTOOLS");   // MSVC++ 10.0 - Visual Studio 2010
#  endif

    if(visualStudioTools != NULL){
      setCompilerEnvScript("\"" + std::string(visualStudioTools) + "..\\..\\VC\\vcvarsall.bat\" " + byteness);
    }
    else{
      std::cout << "WARNING: Visual Studio environment variable not found -> compiler environment (vcvarsall.bat) maybe not correctly setup." << std::endl;
    }
#endif
  }

  template <>
  void device_t<Pthreads>::appendAvailableDevices(std::vector<device> &dList){
    device d;

    d.setup("Pthreads", cpu::getCoreCount(), occa::compact);

    dList.push_back(d);
  }

  template <>
  void device_t<Pthreads>::setCompiler(const std::string &compiler_){
    compiler = compiler_;

    OCCA_EXTRACT_DATA(Pthreads, Device);

    data_.vendor = cpu::compilerVendor(compiler);

    cpu::addSharedBinaryFlagsTo(data_.vendor, compilerFlags);
  }

  template <>
  void device_t<Pthreads>::setCompilerEnvScript(const std::string &compilerEnvScript_){
    compilerEnvScript = compilerEnvScript_;
  }

  template <>
  void device_t<Pthreads>::setCompilerFlags(const std::string &compilerFlags_){
    OCCA_EXTRACT_DATA(Pthreads, Device);

    compilerFlags = compilerFlags_;

    cpu::addSharedBinaryFlagsTo(data_.vendor, compilerFlags);
  }

  template <>
  void device_t<Pthreads>::flush(){}

  template <>
  void device_t<Pthreads>::finish(){
    OCCA_EXTRACT_DATA(Pthreads, Device);

    // Fence local data (incase of out-of-socket updates)
    while(data_.pendingJobs){
#if (OCCA_OS & (LINUX_OS | OSX_OS))
      __asm__ __volatile__ ("lfence");
#else
      __faststorefence(); // NBN: x64 only?
#endif
    }
  }

  template <>
  bool device_t<Pthreads>::fakesUva(){
    return false;
  }

  template <>
  void device_t<Pthreads>::waitFor(streamTag tag){
    finish(); // [-] Not done
  }

  template <>
  stream_t device_t<Pthreads>::createStream(){
    return NULL;
  }

  template <>
  void device_t<Pthreads>::freeStream(stream_t s){}

  template <>
  stream_t device_t<Pthreads>::wrapStream(void *handle_){
    return NULL;
  }

  template <>
  streamTag device_t<Pthreads>::tagStream(){
    streamTag ret;

    ret.tagTime = currentTime();

    return ret;
  }

  template <>
  double device_t<Pthreads>::timeBetween(const streamTag &startTag, const streamTag &endTag){
    return (endTag.tagTime - startTag.tagTime);
  }

  template <>
  kernel_v* device_t<Pthreads>::buildKernelFromSource(const std::string &filename,
                                                      const std::string &functionName,
                                                      const kernelInfo &info_){
    kernel_v *k = new kernel_t<Pthreads>;
    k->dHandle  = this;

    k->buildFromSource(filename, functionName, info_);

    return k;
  }

  template <>
  kernel_v* device_t<Pthreads>::buildKernelFromBinary(const std::string &filename,
                                                      const std::string &functionName){
    kernel_v *k = new kernel_t<Pthreads>;
    k->dHandle  = this;
    k->buildFromBinary(filename, functionName);
    return k;
  }

  template <>
  void device_t<Pthreads>::cacheKernelInLibrary(const std::string &filename,
                                                const std::string &functionName,
                                                const kernelInfo &info_){
    //---[ Creating shared library ]----
    kernel tmpK = occa::device(this).buildKernelFromSource(filename, functionName, info_);
    tmpK.free();

    kernelInfo info = info_;

    addOccaHeadersToInfo(info);

    std::string cachedBinary = getCachedName(filename, getInfoSalt(info));

#if (OCCA_OS & WINDOWS_OS)
    // Windows requires .dll extension
    cachedBinary = cachedBinary + ".dll";
#endif
    //==================================

    library::infoID_t infoID;

    infoID.modelID    = modelID_;
    infoID.kernelName = functionName;

    library::infoHeader_t &header = library::headerMap[infoID];

    header.fileID = -1;
    header.mode   = Pthreads;

    const std::string flatDevID = getIdentifier().flattenFlagMap();

    header.flagsOffset = library::addToScratchPad(flatDevID);
    header.flagsBytes  = flatDevID.size();

    header.contentOffset = library::addToScratchPad(cachedBinary);
    header.contentBytes  = cachedBinary.size();

    header.kernelNameOffset = library::addToScratchPad(functionName);
    header.kernelNameBytes  = functionName.size();
  }

  template <>
  kernel_v* device_t<Pthreads>::loadKernelFromLibrary(const char *cache,
                                                      const std::string &functionName){
    kernel_v *k = new kernel_t<Pthreads>;
    k->dHandle  = this;
    k->loadFromLibrary(cache, functionName);
    return k;
  }

  template <>
  memory_v* device_t<Pthreads>::wrapMemory(void *handle_,
                                           const uintptr_t bytes){
    memory_v *mem = new memory_t<Pthreads>;

    mem->dHandle = this;
    mem->size    = bytes;
    mem->handle  = handle_;

    mem->isAWrapper = true;

    return mem;
  }

  template <>
  memory_v* device_t<Pthreads>::wrapTexture(void *handle_,
                                            const int dim, const occa::dim &dims,
                                            occa::formatType type, const int permissions){
    memory_v *mem = new memory_t<Pthreads>;

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
  memory_v* device_t<Pthreads>::malloc(const uintptr_t bytes,
                                       void *src){
    memory_v *mem = new memory_t<Pthreads>;

    mem->dHandle = this;
    mem->size    = bytes;

    mem->handle = cpu::malloc(bytes);

    if(src != NULL)
      ::memcpy(mem->handle, src, bytes);

    return mem;
  }

  template <>
  memory_v* device_t<Pthreads>::textureAlloc(const int dim, const occa::dim &dims,
                                             void *src,
                                             occa::formatType type, const int permissions){
    memory_v *mem = new memory_t<Pthreads>;

    mem->dHandle = this;
    mem->size    = ((dim == 1) ? dims.x : (dims.x * dims.y)) * type.bytes();

    mem->isTexture = true;
    mem->textureInfo.dim  = dim;

    mem->textureInfo.w = dims.x;
    mem->textureInfo.h = dims.y;
    mem->textureInfo.d = dims.z;

    mem->handle = cpu::malloc(mem->size);

    ::memcpy(mem->handle, src, mem->size);

    mem->textureInfo.arg = mem->handle;
    mem->handle = &(mem->textureInfo);

    return mem;
  }

  template <>
  memory_v* device_t<Pthreads>::mappedAlloc(const uintptr_t bytes,
                                            void *src){
    memory_v *mem = malloc(bytes, src);

    mem->mappedPtr = mem->handle;

    return mem;
  }

  template <>
  void device_t<Pthreads>::free(){
    finish();

    OCCA_EXTRACT_DATA(Pthreads, Device);

    pthread_mutex_destroy( &(data_.pendingJobsMutex) );
    pthread_mutex_destroy( &(data_.kernelMutex) );

    delete (PthreadsDeviceData_t*) data;
  }

  template <>
  int device_t<Pthreads>::simdWidth(){
    simdWidth_ = OCCA_SIMD_WIDTH;
    return OCCA_SIMD_WIDTH;
  }
  //==================================

#include "operators/occaPthreadsKernelOperators.cpp"
};
