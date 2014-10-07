#if OCCA_COI_ENABLED

#if OCCA_OS == WINDOWS_OS
#  error "[COI] Not supported in Windows"
#endif

#include "occaCOI.hpp"

namespace occa {
  //---[ Kernel ]---------------------
  template <>
  kernel_t<COI>::kernel_t(){
    data = NULL;
    dev  = NULL;

    functionName = "";

    dims  = 1;
    inner = occa::dim(1,1,1);
    outer = occa::dim(1,1,1);

    nestedKernelCount = 1;

    preferredDimSize_ = 0;

    startTime = (void*) new double;
    endTime   = (void*) new double;
  }

  template <>
  kernel_t<COI>::kernel_t(const kernel_t<COI> &k){
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

    preferredDimSize_ = k.preferredDimSize_;

    startTime = k.startTime;
    endTime   = k.endTime;
  }

  template <>
  kernel_t<COI>& kernel_t<COI>::operator = (const kernel_t<COI> &k){
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

    preferredDimSize_ = k.preferredDimSize_;

    *((double*) startTime) = *((double*) k.startTime);
    *((double*) endTime)   = *((double*) k.endTime);

    return *this;
  }

  template <>
  kernel_t<COI>::~kernel_t(){}

  template <>
  kernel_t<COI>* kernel_t<COI>::buildFromSource(const std::string &filename,
                                                          const std::string &functionName_,
                                                          const kernelInfo &info_){
    functionName = functionName_;

    kernelInfo info = info_;
    info.addDefine("OCCA_USING_CPU", 1);
    info.addDefine("OCCA_USING_COI", 1);

    info.addOCCAKeywords(occaCOIDefines);

    std::stringstream salt;
    salt << "COI"
         << info.salt()
         << parser::version
         << dev->dHandle->compilerEnvScript
         << dev->dHandle->compiler
         << dev->dHandle->compilerFlags
         << functionName;

    std::string cachedBinary = getCachedName(filename, salt.str());
    std::string libPath, soname;

    getFilePrefixAndName(cachedBinary, libPath, soname);

    std::string libName = "lib" + soname + ".so";

    cachedBinary = libPath + libName;

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

    std::string iCachedBinary = createIntermediateSource(filename,
                                                         cachedBinary,
                                                         info);

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

    const int compileError = system(sCommand.c_str());

    if(compileError){
      releaseFile(cachedBinary);
      throw 1;
    }

    OCCA_EXTRACT_DATA(COI, Kernel);

    COILIBRARY outLibrary;

    const COIRESULT loadingLibraryResult = COIProcessLoadLibraryFromFile(data_.chiefID,
                                                                         cachedBinary.c_str(),
                                                                         soname.c_str(),
                                                                         NULL,
                                                                         &outLibrary);

    if(errorCode != COI_SUCCESS)
      releaseFile(cachedBinary);

    OCCA_COI_CHECK("Kernel: Loading Kernel To Chief", loadingLibraryResult);

    const char *c_functionName = functionName.c_str();

    const COIRESULT getFunctionHandleResult = COIProcessGetFunctionHandles(data_.chiefID,
                                                                           1,
                                                                           &c_functionName,
                                                                           &(data_.kernel));

    if(errorCode != COI_SUCCESS)
      releaseFile(cachedBinary);

    OCCA_COI_CHECK("Kernel: Getting Handle", getFunctionHandleResult);

    releaseFile(cachedBinary);

    return this;
  }

  template <>
  kernel_t<COI>* kernel_t<COI>::buildFromBinary(const std::string &filename,
                                                const std::string &functionName_){
    OCCA_EXTRACT_DATA(COI, Kernel);

    functionName = functionName_;


    std::string libPath, soname;

    getFilePrefixAndName(filename, libPath, soname);

    for(int i = 0; i < soname.size(); ++i){
      if(soname[i] == '.'){
        soname = soname.substr(0, i);
        break;
      }
    }

    COILIBRARY outLibrary;

    OCCA_COI_CHECK("Kernel: Loading Kernel To Chief",
                   COIProcessLoadLibraryFromFile(data_.chiefID,
                                                 filename.c_str(),
                                                 soname.c_str(),
                                                 NULL,
                                                 &outLibrary));

    const char *c_functionName = functionName.c_str();

    OCCA_COI_CHECK("Kernel: Getting Handle",
                   COIProcessGetFunctionHandles(data_.chiefID,
                                                1,
                                                &c_functionName,
                                                &(data_.kernel)));

    return this;
  }

  template <>
  kernel_t<COI>* kernel_t<COI>::loadFromLibrary(const char *cache,
                                                const std::string &functionName_){
    return buildFromBinary(cache, functionName_);
  }

  // [-] Missing
  template <>
  int kernel_t<COI>::preferredDimSize(){
    preferredDimSize_ = 1;

    return preferredDimSize_;
  }

#include "operators/occaCOIKernelOperators.cpp"

  template <>
  double kernel_t<COI>::timeTaken(){
    const double &start = *((double*) startTime);
    const double &end   = *((double*) endTime);

    return 1.0e3*(end - start);
  }

  template <>
  double kernel_t<COI>::timeTakenBetween(void *start, void *end){
    const double &start_ = *((double*) start);
    const double &end_   = *((double*) end);

    return 1.0e3*(end_ - start_);
  }

  template <>
  void kernel_t<COI>::free(){
  }
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<COI>::memory_t(){
    handle = NULL;
    dev    = NULL;
    size = 0;

    isTexture = false;
    textureInfo.arg = NULL;
    textureInfo.dim = 1;
    textureInfo.w = textureInfo.h = textureInfo.d = 0;
  }

  template <>
  memory_t<COI>::memory_t(const memory_t<COI> &m){
    handle = m.handle;
    dev    = m.dev;
    size   = m.size;

    isTexture = m.isTexture;
    textureInfo.arg  = m.textureInfo.arg;
    textureInfo.dim  = m.textureInfo.dim;

    textureInfo.w = m.textureInfo.w;
    textureInfo.h = m.textureInfo.h;
    textureInfo.d = m.textureInfo.d;
  }

  template <>
  memory_t<COI>& memory_t<COI>::operator = (const memory_t<COI> &m){
    handle = m.handle;
    dev    = m.dev;
    size   = m.size;

    isTexture = m.isTexture;
    textureInfo.arg  = m.textureInfo.arg;
    textureInfo.dim  = m.textureInfo.dim;

    textureInfo.w = m.textureInfo.w;
    textureInfo.h = m.textureInfo.h;
    textureInfo.d = m.textureInfo.d;

    return *this;
  }

  template <>
  memory_t<COI>::~memory_t(){}

  template <>
  void* memory_t<COI>::getMemoryHandle(){
    return handle;
  }

  template <>
  void* memory_t<COI>::getTextureHandle(){
    return textureInfo.arg;
  }

  template <>
  void memory_t<COI>::copyFrom(const void *source,
                               const uintptr_t bytes,
                               const uintptr_t offset){
    coiStream &stream = *((coiStream*) dev->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    OCCA_COI_CHECK("Memory: Blocking on Memory Transfer",
                   COIEventWait(1, &(stream.lastEvent),
                                -1, true, NULL, NULL) );

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferWrite(*((coiMemory*) handle),
                                  offset,
                                  source,
                                  bytes_,
                                  COI_COPY_UNSPECIFIED,
                                  false, NULL,
                                  &(stream.lastEvent)));

    OCCA_COI_CHECK("Memory: Blocking on Memory Transfer",
                   COIEventWait(1, &(stream.lastEvent),
                                -1, true, NULL, NULL) );
  }

  template <>
  void memory_t<COI>::copyFrom(const memory_v *source,
                               const uintptr_t bytes,
                               const uintptr_t destOffset,
                               const uintptr_t srcOffset){
    coiStream &stream = *((coiStream*) dev->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + destOffset) <=         size);
    OCCA_CHECK((bytes_ + srcOffset)  <= source->size);

    OCCA_COI_CHECK("Memory: Blocking on Memory Transfer",
                   COIEventWait(1, &(stream.lastEvent),
                                -1, true, NULL, NULL) );

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferCopy(*((coiMemory*) handle),
                                 *((coiMemory*) source->handle),
                                 destOffset,
                                 srcOffset,
                                 bytes_,
                                 COI_COPY_UNSPECIFIED,
                                 false, NULL,
                                 &(stream.lastEvent)));

    OCCA_COI_CHECK("Memory: Blocking on Memory Transfer",
                   COIEventWait(1, &(stream.lastEvent),
                                -1, true, NULL, NULL) );
  }

  template <>
  void memory_t<COI>::copyTo(void *dest,
                             const uintptr_t bytes,
                             const uintptr_t offset){
    coiStream &stream = *((coiStream*) dev->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    OCCA_COI_CHECK("Memory: Blocking on Memory Transfer",
                   COIEventWait(1, &(stream.lastEvent),
                                -1, true, NULL, NULL) );

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferRead(*((coiMemory*) handle),
                                 offset,
                                 dest,
                                 bytes_,
                                 COI_COPY_UNSPECIFIED,
                                 false, NULL,
                                 &(stream.lastEvent)));

    OCCA_COI_CHECK("Memory: Blocking on Memory Transfer",
                   COIEventWait(1, &(stream.lastEvent),
                                -1, true, NULL, NULL) );
  }

  template <>
  void memory_t<COI>::copyTo(memory_v *dest,
                             const uintptr_t bytes,
                             const uintptr_t destOffset,
                             const uintptr_t srcOffset){
    coiStream &stream = *((coiStream*) dev->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + destOffset) <= dest->size);
    OCCA_CHECK((bytes_ + srcOffset)  <=       size);

    OCCA_COI_CHECK("Memory: Blocking on Memory Transfer",
                   COIEventWait(1, &(stream.lastEvent),
                                -1, true, NULL, NULL) );

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferCopy(*((coiMemory*) dest->handle),
                                 *((coiMemory*) handle),
                                 destOffset,
                                 srcOffset,
                                 bytes_,
                                 COI_COPY_UNSPECIFIED,
                                 false, NULL,
                                 &(stream.lastEvent)));

    OCCA_COI_CHECK("Memory: Blocking on Memory Transfer",
                   COIEventWait(1, &(stream.lastEvent),
                                -1, true, NULL, NULL) );
  }

  template <>
  void memory_t<COI>::asyncCopyFrom(const void *source,
                                    const uintptr_t bytes,
                                    const uintptr_t offset){
    coiStream &stream = *((coiStream*) dev->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    OCCA_COI_CHECK("Memory: Blocking on Memory Transfer",
                   COIEventWait(1, &(stream.lastEvent),
                                -1, true, NULL, NULL) );

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferWrite(*((coiMemory*) handle),
                                  offset,
                                  source,
                                  bytes_,
                                  COI_COPY_UNSPECIFIED,
                                  false, NULL,
                                  &(stream.lastEvent)));
  }

  template <>
  void memory_t<COI>::asyncCopyFrom(const memory_v *source,
                                    const uintptr_t bytes,
                                    const uintptr_t destOffset,
                                    const uintptr_t srcOffset){
    coiStream &stream = *((coiStream*) dev->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + destOffset) <=         size);
    OCCA_CHECK((bytes_ + srcOffset)  <= source->size);

    OCCA_COI_CHECK("Memory: Blocking on Memory Transfer",
                   COIEventWait(1, &(stream.lastEvent),
                                -1, true, NULL, NULL) );

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferCopy(*((coiMemory*) handle),
                                 *((coiMemory*) source->handle),
                                 destOffset,
                                 srcOffset,
                                 bytes_,
                                 COI_COPY_UNSPECIFIED,
                                 false, NULL,
                                 &(stream.lastEvent)));
  }

  template <>
  void memory_t<COI>::asyncCopyTo(void *dest,
                                  const uintptr_t bytes,
                                  const uintptr_t offset){
    coiStream &stream = *((coiStream*) dev->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    OCCA_COI_CHECK("Memory: Blocking on Memory Transfer",
                   COIEventWait(1, &(stream.lastEvent),
                                -1, true, NULL, NULL) );

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferRead(*((coiMemory*) handle),
                                 offset,
                                 dest,
                                 bytes_,
                                 COI_COPY_UNSPECIFIED,
                                 false, NULL,
                                 &(stream.lastEvent)));
  }

  template <>
  void memory_t<COI>::asyncCopyTo(memory_v *dest,
                                  const uintptr_t bytes,
                                  const uintptr_t destOffset,
                                  const uintptr_t srcOffset){
    coiStream &stream = *((coiStream*) dev->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + destOffset) <= dest->size);
    OCCA_CHECK((bytes_ + srcOffset)  <=       size);

    OCCA_COI_CHECK("Memory: Blocking on Memory Transfer",
                   COIEventWait(1, &(stream.lastEvent),
                                -1, true, NULL, NULL) );

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferCopy(*((coiMemory*) dest->handle),
                                 *((coiMemory*) handle),
                                 destOffset,
                                 srcOffset,
                                 bytes_,
                                 COI_COPY_UNSPECIFIED,
                                 false, NULL,
                                 &(stream.lastEvent)));
  }

  template <>
  void memory_t<COI>::free(){
    OCCA_COI_CHECK("Memory: free",
                   COIBufferDestroy( *((coiMemory*) handle) ) );

    delete handle;

    size = 0;
  }
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<COI>::device_t() {
    data            = NULL;
    memoryAllocated = 0;

    getEnvironmentVariables();
  }

  template <>
  device_t<COI>::device_t(int platform, int device) {
    data            = NULL;
    memoryAllocated = 0;

    getEnvironmentVariables();
  }

  template <>
  device_t<COI>::device_t(const device_t<COI> &d){
    data            = d.data;
    memoryAllocated = d.memoryAllocated;

    compilerFlags = d.compilerFlags;
  }

  template <>
  device_t<COI>& device_t<COI>::operator = (const device_t<COI> &d){
    data            = d.data;
    memoryAllocated = d.memoryAllocated;

    compilerFlags = d.compilerFlags;

    return *this;
  }

  template <>
  void device_t<COI>::setup(const int device, const int memoryAllocated){
    data = new COIDeviceData_t;

    OCCA_EXTRACT_DATA(COI, Device);

    uint32_t deviceCount;
    OCCA_COI_CHECK("Device: Get Count",
                   COIEngineGetCount(COI_ISA_MIC, &deviceCount));

    OCCA_CHECK(device < deviceCount);

    OCCA_COI_CHECK("Device: Get Handle",
                   COIEngineGetHandle(COI_ISA_MIC, device, &data_.deviceID) );

    std::stringstream salt;
    salt << "COI"
         << occaCOIMain;

    std::string cachedBinary = getCachedName("occaCOIMain", salt.str());

    struct stat buffer;
    bool fileExists = (stat(cachedBinary.c_str(), &buffer) == 0);

    if(fileExists)
      std::cout << "Found cached binary of [occaCOIMain] in [" << cachedBinary << "]\n";
    else{
      //---[ Write File ]-----------------
      std::string prefix, name;

      getFilePrefixAndName(cachedBinary, prefix, name);

      const std::string iCachedBinary = prefix + "i_" + name;

      if(haveFile(cachedBinary)){
        std::cout << "Making [" << iCachedBinary << "]\n";

        std::ofstream fs;
        fs.open(iCachedBinary.c_str());

        fs << occaCOIMain;

        fs.close();

        std::stringstream command;

        command << dev->dHandle->compiler
                << " -o " << cachedBinary
                << " -x c++"
                << ' '    << dev->dHandle->compilerFlags
                << ' '    << iCachedBinary;

        const std::string &sCommand = command.str();

        std::cout << "Compiling [" << functionName << "]\n" << sCommand << "\n\n";

        system(sCommand.c_str());

        releaseFile(cachedBinary);
      }
      else
        waitForFile(cachedBinary);
    }

    // [-] Tentative
    std::string SINK_LD_LIBRARY_PATH;

    char *c_SINK_LD_LIBRARY_PATH = getenv("SINK_LD_LIBRARY_PATH");
    if(c_SINK_LD_LIBRARY_PATH != NULL)
      SINK_LD_LIBRARY_PATH = std::string(c_SINK_LD_LIBRARY_PATH);

    OCCA_COI_CHECK("Device: Initializing",
                   COIProcessCreateFromFile(data_.deviceID,
                                            cachedBinary.c_str(),
                                            0   , NULL,
                                            true, NULL,
                                            true, NULL,
                                            memoryAllocated ? memoryAllocated : (4 << 30), // 4 GB
                                            SINK_LD_LIBRARY_PATH.c_str(),
                                            &(data_.chiefID)) );

    const char *kernelNames[] = {"occaKernelWith1Argument"  , "occaKernelWith2Arguments" , "occaKernelWith3Arguments" ,
                                 "occaKernelWith4Arguments" , "occaKernelWith5Arguments" , "occaKernelWith6Arguments" ,
                                 "occaKernelWith7Arguments" , "occaKernelWith8Arguments" , "occaKernelWith9Arguments" ,
                                 "occaKernelWith10Arguments", "occaKernelWith11Arguments", "occaKernelWith12Arguments",
                                 "occaKernelWith13Arguments", "occaKernelWith14Arguments", "occaKernelWith15Arguments",
                                 "occaKernelWith16Arguments", "occaKernelWith17Arguments", "occaKernelWith18Arguments",
                                 "occaKernelWith19Arguments", "occaKernelWith20Arguments", "occaKernelWith21Arguments",
                                 "occaKernelWith22Arguments", "occaKernelWith23Arguments", "occaKernelWith24Arguments",
                                 "occaKernelWith25Arguments"};

    // [-] More hard-coding, if you know what I mean
    OCCA_COI_CHECK("Device: Getting Kernel Wrappers",
                   COIProcessGetFunctionHandles(data_.chiefID,
                                                25,
                                                kernelNames,
                                                data_.kernelWrapper));
  }

  template <>
  deviceIdentifier device_t<COI>::getIdentifier() const {
    deviceIdentifier dID;

    dID.mode_ = COI;

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
  void device_t<COI>::getEnvironmentVariables(){
    const char *c_compiler = getenv("OCCA_COI_COMPILER");

    if(c_compiler != NULL)
      compiler = std::string(c_compiler);
    else
      compiler = "icpc";

    const char *c_compilerFlags = getenv("OCCA_COI_COMPILER_FLAGS");

    if(c_compilerFlags != NULL)
      compilerFlags = std::string(c_compilerFlags);
    else{
#if OCCA_DEBUG_ENABLED
      compilerFlags = "-g";
#else
      compilerFlags = "-O3";
#endif
    }
  }

  template <>
  void device_t<COI>::appendAvailableDevices(std::vector<device> &dList){
    uint32_t deviceCount;
    OCCA_COI_CHECK("Device: Get Count",
                   COIEngineGetCount(COI_ISA_MIC, &deviceCount));

    for(int i = 0; i < deviceCount; ++i){
      device d;
      d.setup("COI", i);

      dList.push_back(d);
    }
  }

  template <>
  void device_t<COI>::setCompiler(const std::string &compiler_){
    compiler = compiler_;
  }

  template <>
  void device_t<COI>::setCompilerEnvScript(const std::string &compilerEnvScript_){
    compilerEnvScript = compilerEnvScript_;
  }

  template <>
  void device_t<COI>::setCompilerFlags(const std::string &compilerFlags_){
    compilerFlags = compilerFlags_;
  }

  template <>
  std::string& device_t<COI>::getCompiler(){
    return compiler;
  }

  template <>
  std::string& device_t<COI>::getCompilerEnvScript(){
    return compilerEnvScript;
  }

  template <>
  std::string& device_t<COI>::getCompilerFlags(){
    return compilerFlags;
  }

  template <>
  void device_t<COI>::flush(){}

  template <>
  void device_t<COI>::finish(){
    coiStream &stream = *((coiStream*) dev->currentStream);

    OCCA_COI_CHECK("Device: Waiting for Event",
                   COIEventWait(1, &(stream.lastEvent),
                                -1, true, NULL, NULL) );
  }

  template <>
  stream device_t<COI>::genStream(){
    OCCA_EXTRACT_DATA(COI, Device);

    coiStream *retStream = new coiStream;

    OCCA_COI_CHECK("Device: Generating a Stream",
                   COIPipelineCreate(data_.chiefID,
                                     NULL, 0,
                                     &(retStream->handle)) );

    return retStream;
  }

  template <>
  void device_t<COI>::freeStream(stream s){
    if(s == NULL)
      return;

    coiStream *stream = (coiStream*) s;

    OCCA_COI_CHECK("Device: Freeing a Stream",
                   COIPipelineDestroy(stream->handle));

    delete stream;
  }

  // [-] Event-based timing in COI?
  template <>
  tag device_t<COI>::tagStream(){
    tag ret;

    ret.tagTime = 0;

    return ret;
  }

  template <>
  double device_t<COI>::timeBetween(const tag &startTag, const tag &endTag){
    return (endTag.tagTime - startTag.tagTime);
  }

  template <>
  kernel_v* device_t<COI>::buildKernelFromSource(const std::string &filename,
                                                 const std::string &functionName,
                                                 const kernelInfo &info_){
    OCCA_EXTRACT_DATA(COI, Device);

    kernel_v *k = new kernel_t<COI>;

    k->dev  = dev;
    k->data = new COIKernelData_t;

    COIKernelData_t &kData_ = *((COIKernelData_t*) k->data);

    kData_.chiefID = data_.chiefID;

    k->buildFromSource(filename, functionName, info_);
    return k;
  }

  template <>
  kernel_v* device_t<COI>::buildKernelFromBinary(const std::string &filename,
                                                 const std::string &functionName){
    OCCA_EXTRACT_DATA(COI, Device);

    kernel_v *k = new kernel_t<COI>;

    k->dev  = dev;
    k->data = new COIKernelData_t;

    COIKernelData_t &kData_ = *((COIKernelData_t*) k->data);

    kData_.chiefID = data_.chiefID;

    k->buildFromBinary(filename, functionName);
    return k;
  }

  template <>
  void device_t<COI>::cacheKernelInLibrary(const std::string &filename,
                                           const std::string &functionName,
                                           const kernelInfo &info_){
    //---[ Creating shared library ]----
    kernel tmpK = dev->buildKernelFromSource(filename, functionName, info_);
    tmpK.free();

    kernelInfo info = info_;
    info.addDefine("OCCA_USING_CPU", 1);
    info.addDefine("OCCA_USING_COI", 1);

    info.addOCCAKeywords(occaCOIDefines);

    std::stringstream salt;

    salt << "COI"
         << info.salt()
         << parser::version
         << compilerEnvScript
         << compiler
         << compilerFlags
         << functionName;

    std::string cachedBinary = getCachedName(filename, salt.str());
    std::string libPath, soname;

    getFilePrefixAndName(cachedBinary, libPath, soname);

    std::string libName = "lib" + soname + ".so";

    cachedBinary = libPath + libName;
    //==================================

    library::infoID_t infoID;

    infoID.modelID    = dev->modelID_;
    infoID.kernelName = functionName;

    library::infoHeader_t &header = library::headerMap[infoID];

    header.fileID = -1;
    header.mode   = COI;

    const std::string flatDevID = getIdentifiers().flattenFlagMap();

    header.flagsOffset = library::addToScratchPad(flatDevID);
    header.flagsBytes  = flatDevID.size();

    header.contentOffset = library::addToScratchPad(cachedBinary);
    header.contentBytes  = cachedBinary.size();

    header.kernelNameOffset = library::addToScratchPad(functionName);
    header.kernelNameBytes  = functionName.size();
  }

  template <>
  kernel_v* device_t<COI>::loadKernelFromLibrary(const char *cache,
                                                 const std::string &functionName_){
    OCCA_EXTRACT_DATA(COI, Device);

    kernel_v *k = new kernel_t<COI>;

    k->dev  = dev;
    k->data = new COIKernelData_t;

    COIKernelData_t &kData_ = *((COIKernelData_t*) k->data);

    kData_.chiefID = data_.chiefID;

    k->loadFromLibrary(cache, functionName_);
    return k;
  }

  template <>
  memory_v* device_t<COI>::wrapMemory(void *handle_,
                                      const uintptr_t bytes){
    memory_v *mem = new memory_t<COI>;

    mem->dev    = dev;
    mem->size   = bytes;
    mem->handle = handle_;

    return mem;
  }

  template <>
  memory_v* device_t<COI>::wrapTexture(void *handle_,
                                       const int dim, const occa::dim &dims,
                                       occa::formatType type, const int permissions){
#warning "Textures not supported in COI yet"

    memory_v *mem = new memory_t<COI>;
    return mem;
  }

  template <>
  memory_v* device_t<COI>::malloc(const uintptr_t bytes,
                                  void *source){
    OCCA_EXTRACT_DATA(COI, Device);

    memory_v *mem = new memory_t<COI>;

    mem->dev    = dev;
    mem->handle = new coiMemory;
    mem->size   = bytes;

    OCCA_COI_CHECK("Device: Malloc",
                   COIBufferCreate(bytes,
                                   COI_BUFFER_NORMAL,
                                   0,
                                   source,
                                   1,
                                   &(data_.chiefID),
                                   (coiMemory*) mem->handle) );

    return mem;
  }

  template <>
  memory_v* device_t<COI>::talloc(const int dim, const occa::dim &dims,
                                  void *source,
                                  occa::formatType type, const int permissions){
#warning "Textures not supported in COI yet"

    memory_v *mem = new memory_t<COI>;

    mem->dev  = dev;
    mem->size = ((dim == 1) ? dims.x : (dims.x * dims.y)) * type.bytes();

    mem->isTexture = true;
    mem->textureInfo.dim  = dim;
    mem->textureInfo.dims = dims;

#if   OCCA_OS == LINUX_OS
    posix_memalign(&mem->handle, OCCA_MEM_ALIGN, mem->size);
#elif OCCA_OS == OSX_OS
    mem->handle = ::malloc(mem->size);
#else
#  warning "Aligned memory not supported in Windows yet"
    mem->handle = ::malloc(mem->size);
#endif

    ::memcpy(mem->handle, source, mem->size);

    mem->textureInfo.arg = mem->handle;
    mem->handle = &(mem->textureInfo);

    return mem;
  }

  template <>
  void device_t<COI>::free(){
    OCCA_EXTRACT_DATA(COI, Device);

    OCCA_COI_CHECK("Device: Freeing Chief Processes",
                   COIProcessDestroy(data_.chiefID,
                                     -1,
                                     false,
                                     NULL,
                                     NULL));

    delete data;
  }

  template <>
  int device_t<COI>::simdWidth(){
    simdWidth_ = 16; // [-] AVX-512

    return 16;
  }
  //==================================


  //---[ Error Handling ]-------------
  std::string coiError(coiStatus e){
    return std::string( COIResultGetName(e) );
  }
  //==================================
};

#endif
