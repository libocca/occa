#include "occaOpenMP.hpp"

namespace occa {
  //---[ Kernel ]---------------------
  template <>
  kernel_t<OpenMP>::kernel_t(){
    data = NULL;
    dev  = NULL;

    functionName = "";

    dims  = 1;
    inner = occa::dim(1,1,1);
    outer = occa::dim(1,1,1);

    nestedKernelCount = 1;

    nestedKernelCount = 1;
    setDimsKernels    = NULL;
    nestedKernels     = NULL;

    startTime = (void*) new double;
    endTime   = (void*) new double;
  }

  template <>
  kernel_t<OpenMP>::kernel_t(const kernel_t<OpenMP> &k){
    data = k.data;
    dev  = k.dev;

    functionName = k.functionName;

    dims  = k.dims;
    inner = k.inner;
    outer = k.outer;

    nestedKernelCount = k.nestedKernelCount;
    // setDimsKernels = new kernel_v*[nestedKernelCount];
    nestedKernels  = new kernel_v*[nestedKernelCount];

    for(int i = 0; i < nestedKernelCount; ++i)
      nestedKernels[i] = k.nestedKernels[i];

    nestedKernelCount = k.nestedKernelCount;
    setDimsKernels    = k.setDimsKernels;
    nestedKernels     = k.nestedKernels;

    startTime = k.startTime;
    endTime   = k.endTime;
  }

  template <>
  kernel_t<OpenMP>& kernel_t<OpenMP>::operator = (const kernel_t<OpenMP> &k){
    data = k.data;
    dev  = k.dev;

    functionName = k.functionName;

    dims  = k.dims;
    inner = k.inner;
    outer = k.outer;

    nestedKernelCount = k.nestedKernelCount;
    // setDimsKernels = new kernel_v*[nestedKernelCount];
    nestedKernels  = new kernel_v*[nestedKernelCount];

    for(int i = 0; i < nestedKernelCount; ++i)
      nestedKernels[i] = k.nestedKernels[i];

    nestedKernelCount = k.nestedKernelCount;
    setDimsKernels    = k.setDimsKernels;
    nestedKernels     = k.nestedKernels;

    *((double*) startTime) = *((double*) k.startTime);
    *((double*) endTime)   = *((double*) k.endTime);

    return *this;
  }

  template <>
  kernel_t<OpenMP>::~kernel_t(){}

  template <>
  kernel_t<OpenMP>* kernel_t<OpenMP>::buildFromSource(const std::string &filename,
                                                      const std::string &functionName_,
                                                      const kernelInfo &info_){
    functionName = functionName_;

    kernelInfo info = info_;
    info.addDefine("OCCA_USING_CPU"   , 1);
    info.addDefine("OCCA_USING_OPENMP", 1);

#if OCCA_OPENMP_ENABLED
    info.addIncludeDefine("omp.h");
#endif

    info.addOCCAKeywords(occaOpenMPDefines);

    std::stringstream salt;
    salt << "OpenMP"
         << info.salt()
         << parser::version
         << dev->dHandle->compilerEnvScript
         << dev->dHandle->compiler
         << dev->dHandle->compilerFlags
         << functionName;

    std::string cachedBinary = getCachedName(filename, salt.str());

#if OCCA_OS == WINDOWS_OS
    // Windows refuses to load dll's that do not end with '.dll'
    cachedBinary = cachedBinary + ".dll";
#endif

    struct stat buffer;
    bool fileExists = (stat(cachedBinary.c_str(), &buffer) == 0);

    if(fileExists){
      std::cout << "Found cached binary of [" << filename << "] in [" << cachedBinary << "]\n";
      return buildFromBinary(cachedBinary, functionName);
    }

    if(!haveFile(cachedBinary)){
      waitForFile(cachedBinary);

      return buildFromBinary(cachedBinary, functionName);
    }

    data = new OpenMPKernelData_t;

    //---[ Create Intermediate ]--------
    std::string prefix, name;
    getFilePrefixAndName(cachedBinary, prefix, name);

    std::string extension = getFileExtension(filename);

    const std::string iCachedBinary = prefix + "i_" + name;
    if(extension == "okl"){
      const std::string pCachedBinary = prefix + "p_" + name;
      parser fileParser;

      std::ofstream fs;
      fs.open(pCachedBinary.c_str());

      fs << info.header << readFile(filename);

      fs.close();

      fs.open(iCachedBinary.c_str());
      fs << info.occaKeywords << fileParser.parseFile(pCachedBinary);

      fs.close();

      kernelInfoIterator kIt = fileParser.kernelInfoMap.find(functionName);

      if(kIt != fileParser.kernelInfoMap.end()){
        parserNamespace::kernelInfo &kInfo = *((kIt++)->second);

        nestedKernelCount = kInfo.nestedKernels.size();

        if(nestedKernelCount != 1){
          std::stringstream ss;
          nestedKernels = new kernel_v*[nestedKernelCount];

          releaseFile(cachedBinary);

          for(int k = 0; k < nestedKernelCount; ++k){
            ss.str("");
            ss << k;

            nestedKernels[k] = dev->dHandle->buildKernelFromSource(filename,
                                                                   kInfo.baseName + ss.str(),
                                                                   info_);
          }

          return this;
        }
      }
    }
    else{
      std::ofstream fs;
      fs.open(iCachedBinary.c_str());

      fs << info.occaKeywords << info.header << readFile(filename);

      fs.close();
    }
    //==================================

    std::stringstream command;

    if(dev->dHandle->compilerEnvScript.size())
      command << dev->dHandle->compilerEnvScript << " && ";

    command << dev->dHandle->compiler
#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
            << " -x c++ -w -fPIC -shared"
#else
            << " /TP /LD /D MC_CL_EXE"
#endif
            << ' '    << dev->dHandle->compilerFlags
            << ' '    << info.flags
            << ' '    << iCachedBinary
#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
            << " -o " << cachedBinary
#else
            << " /link /OUT:" << cachedBinary
#endif
            << std::endl;

    const std::string &sCommand = command.str();

    std::cout << "Compiling [" << functionName << "]\n" << sCommand << "\n";

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
    const int compileError = system(sCommand.c_str());
#else
    const int compileError = system(("\"" +  sCommand + "\"").c_str());
#endif

    if(compileError){
      releaseFile(cachedBinary);
      throw 1;
    }

    OCCA_EXTRACT_DATA(OpenMP, Kernel);

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
    data_.dlHandle = dlopen(cachedBinary.c_str(), RTLD_NOW);

    if(data_.dlHandle == NULL){
      releaseFile(cachedBinary);
      throw 1;
    }
#else
    data_.dlHandle = LoadLibraryA(cachedBinary.c_str());

    if(data_.dlHandle == NULL) {
      DWORD errCode = GetLastError();
      std::cerr << "Unable to load dll: " << cachedBinary << " (WIN32 error code: " << errCode << ")" << std::endl;

      throw 1;
    }
#endif

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
    data_.handle = dlsym(data_.dlHandle, functionName.c_str());

    char *dlError;
    if ((dlError = dlerror()) != NULL)  {
      fputs(dlError, stderr);
      releaseFile(cachedBinary);
      throw 1;
    }
#else
    data_.handle = GetProcAddress((HMODULE) (data_.dlHandle), functionName.c_str());

    if(data_.dlHandle == NULL) {
      fputs("unable to load function", stderr);
      releaseFile(cachedBinary);
      throw 1;
    }
#endif

    releaseFile(cachedBinary);

    return this;
  }

  template <>
  kernel_t<OpenMP>* kernel_t<OpenMP>::buildFromBinary(const std::string &filename,
                                                      const std::string &functionName_){
    data = new OpenMPKernelData_t;

    OCCA_EXTRACT_DATA(OpenMP, Kernel);

    functionName = functionName_;

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
    data_.dlHandle = dlopen(filename.c_str(), RTLD_LAZY | RTLD_LOCAL);
#else
    data_.dlHandle = LoadLibraryA(filename.c_str());
    if(data_.dlHandle == NULL) {
      DWORD errCode = GetLastError();
      std::cerr << "Unable to load dll: " << filename << " (WIN32 error code: " << errCode << ")" << std::endl;
      throw 1;
    }
#endif
    OCCA_CHECK(data_.dlHandle != NULL);

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
    data_.handle = dlsym(data_.dlHandle, functionName.c_str());

    char *dlError;
    if ((dlError = dlerror()) != NULL)  {
      fputs(dlError, stderr);
      throw 1;
    }
#else
    data_.handle = GetProcAddress((HMODULE) (data_.dlHandle), functionName.c_str());
    if(data_.dlHandle == NULL) {
      fputs("unable to load function", stderr);
      throw 1;
    }
#endif

    return this;
  }

  template <>
  kernel_t<OpenMP>* kernel_t<OpenMP>::loadFromLibrary(const char *cache,
                                                      const std::string &functionName_){
    return buildFromBinary(cache, functionName_);
  }

  // [-] Missing
  template <>
  int kernel_t<OpenMP>::preferredDimSize(){
    preferredDimSize_ = OCCA_SIMD_WIDTH;
    return OCCA_SIMD_WIDTH;
  }

#include "operators/occaOpenMPKernelOperators.cpp"

  template <>
  double kernel_t<OpenMP>::timeTaken(){
    const double &start = *((double*) startTime);
    const double &end   = *((double*) endTime);

    return 1.0e3*(end - start);
  }

  template <>
  double kernel_t<OpenMP>::timeTakenBetween(void *start, void *end){
    const double &start_ = *((double*) start);
    const double &end_   = *((double*) end);

    return 1.0e3*(end_ - start_);
  }

  template <>
  void kernel_t<OpenMP>::free(){
    // [-] Fix later
    OCCA_EXTRACT_DATA(OpenMP, Kernel);

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
    dlclose(data_.dlHandle);
#else
    FreeLibrary((HMODULE) (data_.dlHandle));
#endif
  }
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<OpenMP>::memory_t(){
    handle = NULL;
    dev    = NULL;
    size = 0;

    isTexture = false;
    textureInfo.arg = NULL;
    textureInfo.dim = 1;
    textureInfo.w = textureInfo.h = textureInfo.d = 0;

    isAWrapper = false;
  }

  template <>
  memory_t<OpenMP>::memory_t(const memory_t<OpenMP> &m){
    handle = m.handle;
    dev    = m.dev;
    size   = m.size;

    isTexture = m.isTexture;
    textureInfo.arg  = m.textureInfo.arg;
    textureInfo.dim  = m.textureInfo.dim;

    textureInfo.w = m.textureInfo.w;
    textureInfo.h = m.textureInfo.h;
    textureInfo.d = m.textureInfo.d;

    if(isTexture)
      handle = &textureInfo;

    isAWrapper = m.isAWrapper;
  }

  template <>
  memory_t<OpenMP>& memory_t<OpenMP>::operator = (const memory_t<OpenMP> &m){
    handle = m.handle;
    dev    = m.dev;
    size   = m.size;

    isTexture = m.isTexture;
    textureInfo.arg  = m.textureInfo.arg;
    textureInfo.dim  = m.textureInfo.dim;

    textureInfo.w = m.textureInfo.w;
    textureInfo.h = m.textureInfo.h;
    textureInfo.d = m.textureInfo.d;

    if(isTexture)
      handle = &textureInfo;

    isAWrapper = m.isAWrapper;

    return *this;
  }

  template <>
  memory_t<OpenMP>::~memory_t(){}

  template <>
  void* memory_t<OpenMP>::getMemoryHandle(){
    return handle;
  }

  template <>
  void* memory_t<OpenMP>::getTextureHandle(){
    return textureInfo.arg;
  }

  template <>
  void memory_t<OpenMP>::copyFrom(const void *source,
                                  const uintptr_t bytes,
                                  const uintptr_t offset){
    dev->finish();

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    void *destPtr      = ((char*) (isTexture ? textureInfo.arg : handle)) + offset;
    const void *srcPtr = source;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<OpenMP>::copyFrom(const memory_v *source,
                                  const uintptr_t bytes,
                                  const uintptr_t destOffset,
                                  const uintptr_t srcOffset){
    dev->finish();

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + destOffset) <= size);
    OCCA_CHECK((bytes_ + srcOffset)  <= source->size);

    void *destPtr      = ((char*) (isTexture         ? textureInfo.arg         : handle))         + destOffset;
    const void *srcPtr = ((char*) (source->isTexture ? source->textureInfo.arg : source->handle)) + srcOffset;;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<OpenMP>::copyTo(void *dest,
                                const uintptr_t bytes,
                                const uintptr_t offset){
    dev->finish();

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    void *destPtr      = dest;
    const void *srcPtr = ((char*) (isTexture ? textureInfo.arg : handle)) + offset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<OpenMP>::copyTo(memory_v *dest,
                                const uintptr_t bytes,
                                const uintptr_t destOffset,
                                const uintptr_t srcOffset){
    dev->finish();

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + srcOffset)  <= size);
    OCCA_CHECK((bytes_ + destOffset) <= dest->size);

    void *destPtr      = ((char*) (dest->isTexture ? dest->textureInfo.arg : dest->handle)) + destOffset;
    const void *srcPtr = ((char*) (isTexture ? textureInfo.arg : handle))       + srcOffset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<OpenMP>::asyncCopyFrom(const void *source,
                                       const uintptr_t bytes,
                                       const uintptr_t offset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    void *destPtr      = ((char*) (isTexture ? textureInfo.arg : handle)) + offset;
    const void *srcPtr = source;


    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<OpenMP>::asyncCopyFrom(const memory_v *source,
                                       const uintptr_t bytes,
                                       const uintptr_t destOffset,
                                       const uintptr_t srcOffset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + destOffset) <= size);
    OCCA_CHECK((bytes_ + srcOffset)  <= source->size);

    void *destPtr      = ((char*) (isTexture         ? textureInfo.arg         : handle))         + destOffset;
    const void *srcPtr = ((char*) (source->isTexture ? source->textureInfo.arg : source->handle)) + srcOffset;;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<OpenMP>::asyncCopyTo(void *dest,
                                     const uintptr_t bytes,
                                     const uintptr_t offset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    void *destPtr      = dest;
    const void *srcPtr = ((char*) (isTexture ? textureInfo.arg : handle)) + offset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<OpenMP>::asyncCopyTo(memory_v *dest,
                                     const uintptr_t bytes,
                                     const uintptr_t destOffset,
                                     const uintptr_t srcOffset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + srcOffset)  <= size);
    OCCA_CHECK((bytes_ + destOffset) <= dest->size);

    void *destPtr      = ((char*) (dest->isTexture ? dest->textureInfo.arg : dest->handle)) + destOffset;
    const void *srcPtr = ((char*) (isTexture ? textureInfo.arg : handle))       + srcOffset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<OpenMP>::free(){
    if(isTexture)
      ::free(textureInfo.arg);
    else
      ::free(handle);

    size = 0;
  }
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<OpenMP>::device_t(){
    data            = NULL;
    memoryAllocated = 0;

    getEnvironmentVariables();
  }

  template <>
  device_t<OpenMP>::device_t(int platform, int device){
    data            = NULL;
    memoryAllocated = 0;

    getEnvironmentVariables();
  }

  template <>
  device_t<OpenMP>::device_t(const device_t<OpenMP> &d){
    data            = d.data;
    memoryAllocated = d.memoryAllocated;

    compiler      = d.compiler;
    compilerFlags = d.compilerFlags;
  }

  template <>
  device_t<OpenMP>& device_t<OpenMP>::operator = (const device_t<OpenMP> &d){
    data            = d.data;
    memoryAllocated = d.memoryAllocated;

    compiler      = d.compiler;
    compilerFlags = d.compilerFlags;

    return *this;
  }

  template <>
  void device_t<OpenMP>::setup(const int unusedArg1, const int unusedArg2){}

  template <>
  deviceIdentifier device_t<OpenMP>::getIdentifier() const {
    deviceIdentifier dID;

    dID.mode_ = OpenMP;

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
  void device_t<OpenMP>::getEnvironmentVariables(){
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
      compilerFlags = "-D__extern_always_inline=inline -O3";
#  endif
    }
#else
#  if OCCA_DEBUG_ENABLED
    compilerFlags = " /Od ";
#  else
    compilerFlags = " /Ox /openmp ";
#  endif
    std::string byteness;
    if(sizeof(void*) == 4)
      byteness = "x86 ";
    else if(sizeof(void*) == 8)
      byteness = "amd64";
    else
      throw 1;

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
  void device_t<OpenMP>::appendAvailableDevices(std::vector<device> &dList){
    device d;
    d.setup("OpenMP");

    dList.push_back(d);
  }

  template <>
  void device_t<OpenMP>::setCompiler(const std::string &compiler_){
    compiler = compiler_;
  }

  template <>
  void device_t<OpenMP>::setCompilerEnvScript(const std::string &compilerEnvScript_){
    compilerEnvScript = compilerEnvScript_;
  }

  template <>
  void device_t<OpenMP>::setCompilerFlags(const std::string &compilerFlags_){
    compilerFlags = compilerFlags_;
  }

  template <>
  std::string& device_t<OpenMP>::getCompiler(){
    return compiler;
  }

  template <>
  std::string& device_t<OpenMP>::getCompilerEnvScript(){
    return compilerEnvScript;
  }

  template <>
  std::string& device_t<OpenMP>::getCompilerFlags(){
    return compilerFlags;
  }

  template <>
  void device_t<OpenMP>::flush(){}

  template <>
  void device_t<OpenMP>::finish(){}

  template <>
  stream device_t<OpenMP>::genStream(){
    return NULL;
  }

  template <>
  void device_t<OpenMP>::freeStream(stream s){}

  template <>
  stream device_t<OpenMP>::wrapStream(void *handle_){
    return NULL;
  }

  template <>
  tag device_t<OpenMP>::tagStream(){
    tag ret;

    ret.tagTime = currentTime();

    return ret;
  }

  template <>
  double device_t<OpenMP>::timeBetween(const tag &startTag, const tag &endTag){
    return (endTag.tagTime - startTag.tagTime);
  }

  template <>
  kernel_v* device_t<OpenMP>::buildKernelFromSource(const std::string &filename,
                                                    const std::string &functionName,
                                                    const kernelInfo &info_){
    kernel_v *k = new kernel_t<OpenMP>;
    k->dev = dev;
    k->buildFromSource(filename, functionName, info_);
    return k;
  }

  template <>
  kernel_v* device_t<OpenMP>::buildKernelFromBinary(const std::string &filename,
                                                    const std::string &functionName){
    kernel_v *k = new kernel_t<OpenMP>;
    k->dev = dev;
    k->buildFromBinary(filename, functionName);
    return k;
  }

  template <>
  void device_t<OpenMP>::cacheKernelInLibrary(const std::string &filename,
                                              const std::string &functionName,
                                              const kernelInfo &info_){
    //---[ Creating shared library ]----
    kernel tmpK = dev->buildKernelFromSource(filename, functionName, info_);
    tmpK.free();

    kernelInfo info = info_;
    info.addDefine("OCCA_USING_CPU"   , 1);
    info.addDefine("OCCA_USING_OPENMP", 1);

#if OCCA_OPENMP_ENABLED
    info.addIncludeDefine("omp.h");
#endif

    info.addOCCAKeywords(occaOpenMPDefines);

    std::stringstream salt;
    salt << "OpenMP"
         << info.salt()
         << parser::version
         << compilerEnvScript
         << compiler
         << compilerFlags
         << functionName;

    std::string cachedBinary = getCachedName(filename, salt.str());

#if OCCA_OS == WINDOWS_OS
    // Windows refuses to load dll's that do not end with '.dll'
    cachedBinary = cachedBinary + ".dll";
#endif
    //==================================

    library::infoID_t infoID;

    infoID.modelID    = dev->modelID_;
    infoID.kernelName = functionName;

    library::infoHeader_t &header = library::headerMap[infoID];

    header.fileID = -1;
    header.mode   = OpenMP;

    const std::string flatDevID = getIdentifier().flattenFlagMap();

    header.flagsOffset = library::addToScratchPad(flatDevID);
    header.flagsBytes  = flatDevID.size();

    header.contentOffset = library::addToScratchPad(cachedBinary);
    header.contentBytes  = cachedBinary.size();

    header.kernelNameOffset = library::addToScratchPad(functionName);
    header.kernelNameBytes  = functionName.size();
  }

  template <>
  kernel_v* device_t<OpenMP>::loadKernelFromLibrary(const char *cache,
                                                    const std::string &functionName_){
    kernel_v *k = new kernel_t<OpenMP>;
    k->dev = dev;
    k->loadFromLibrary(cache, functionName_);
    return k;
  }

  template <>
  memory_v* device_t<OpenMP>::wrapMemory(void *handle_,
                                         const uintptr_t bytes){
    memory_v *mem = new memory_t<OpenMP>;

    mem->dev    = dev;
    mem->size   = bytes;
    mem->handle = handle_;

    mem->isAWrapper = true;

    return mem;
  }

  template <>
  memory_v* device_t<OpenMP>::wrapTexture(void *handle_,
                                          const int dim, const occa::dim &dims,
                                          occa::formatType type, const int permissions){
    memory_v *mem = new memory_t<OpenMP>;

    mem->dev  = dev;
    mem->size = ((dim == 1) ? dims.x : (dims.x * dims.y)) * type.bytes();

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
  memory_v* device_t<OpenMP>::malloc(const uintptr_t bytes,
                                     void *source){
    memory_v *mem = new memory_t<OpenMP>;

    mem->dev  = dev;
    mem->size = bytes;

#if   OCCA_OS == LINUX_OS
    posix_memalign(&mem->handle, OCCA_MEM_ALIGN, bytes);
#elif OCCA_OS == OSX_OS
    mem->handle = ::malloc(bytes);
#else
    mem->handle = ::malloc(bytes);
#endif

    if(source != NULL)
      ::memcpy(mem->handle, source, bytes);

    return mem;
  }

  template <>
  memory_v* device_t<OpenMP>::talloc(const int dim, const occa::dim &dims,
                                     void *source,
                                     occa::formatType type, const int permissions){
    memory_v *mem = new memory_t<OpenMP>;

    mem->dev  = dev;
    mem->size = ((dim == 1) ? dims.x : (dims.x * dims.y)) * type.bytes();

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

    ::memcpy(mem->textureInfo.arg, source, mem->size);

    std::cout << "Allocating: [" << (void*) mem->textureInfo.arg << "]\n";

    mem->handle = &(mem->textureInfo);

    return mem;
  }

  template <>
  void device_t<OpenMP>::free(){}

  template <>
  int device_t<OpenMP>::simdWidth(){
    simdWidth_ = OCCA_SIMD_WIDTH;
    return OCCA_SIMD_WIDTH;
  }
  //==================================
};
