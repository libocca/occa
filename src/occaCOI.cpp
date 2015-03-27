#if OCCA_COI_ENABLED

#if OCCA_OS == WINDOWS_OS
#  error "[COI] Not supported in Windows"
#endif

#include "occaCOI.hpp"

namespace occa {
  //---[ Helper Functions ]-----------
  namespace coi {
    void initDevice(COIDeviceData_t &data){
      std::stringstream salt;

      salt << "COI"
           << occaCOIMain;

      std::string cachedBinary = getCachedName("occaCOIMain", salt.str());

      struct stat buffer;
      bool fileExists = (stat(cachedBinary.c_str(), &buffer) == 0);

      if(fileExists){
        if(verboseCompilation_f)
          std::cout << "Found cached binary of [occaCOIMain] in [" << cachedBinary << "]\n";
      }
      else{
        //---[ Write File ]-----------------
        std::string prefix, name;

        getFilePrefixAndName(cachedBinary, prefix, name);

        const std::string iCachedBinary = prefix + "i_" + name;

        if(haveFile(cachedBinary)){
          if(verboseCompilation_f)
            std::cout << "Making [" << iCachedBinary << "]\n";

          std::ofstream fs;
          fs.open(iCachedBinary.c_str());

          fs << occaCOIMain;

          fs.close();

          std::stringstream command;

          command << dHandle->compiler
                  << " -o " << cachedBinary
                  << " -x c++"
                  << ' '    << dHandle->compilerFlags
                  << ' '    << iCachedBinary;

          const std::string &sCommand = command.str();

          if(verboseCompilation_f)
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
                     COIProcessCreateFromFile(data.deviceID,
                                              cachedBinary.c_str(),
                                              0   , NULL,
                                              true, NULL,
                                              true, NULL,
                                              bytesAllocated ? bytesAllocated : (4 << 30), // 4 GB
                                              SINK_LD_LIBRARY_PATH.c_str(),
                                              &(data.chiefID)) );

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
                     COIProcessGetFunctionHandles(data.chiefID,
                                                  25,
                                                  kernelNames,
                                                  data.kernelWrapper));
    }

    std::string getDeviceListInfo(){
      std::stringstream ss;

      uint32_t deviceCount;

      OCCA_COI_CHECK("Device: Get Count",
                     COIEngineGetCount(COI_ISA_MIC, &deviceCount));

      // << "==============o=======================o==========================================\n";
      ss << "     COI      |  Xeon Phi Count       | " << deviceCount                     << '\n';
      // << "==============o=======================o==========================================\n";

      return ss.str();
    }

    occa::device wrapDevice(COIENGINE device){
      occa::device dev;
      device_t<COI> &dHandle   = *(new device_t<COI>());
      COIDeviceData_t &devData = *(new COIDeviceData_t);

      dev.strMode = "COI";
      dev.dHandle = &dHandle;

      //---[ Setup ]----------
      dHandle.data = &devData;

      devData.deviceID = device;

      coi::initDevice(devData);
      //======================

      dHandle.modelID_ = library::deviceModelID(dHandle.getIdentifier());
      dHandle.id_      = library::genDeviceID();

      dHandle.currentStream = dHandle.createStream();

      return dev;
    }
  };
  //==================================


  //---[ Kernel ]---------------------
  template <>
  kernel_t<COI>::kernel_t(){
    data    = NULL;
    dHandle = NULL;

    dims  = 1;
    inner = occa::dim(1,1,1);
    outer = occa::dim(1,1,1);

    nestedKernelCount = 0;

    preferredDimSize_ = 0;
  }

  template <>
  kernel_t<COI>::kernel_t(const kernel_t<COI> &k){
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

    preferredDimSize_ = k.preferredDimSize_;
  }

  template <>
  kernel_t<COI>& kernel_t<COI>::operator = (const kernel_t<COI> &k){
    data    = k.data;
    dhandle = k.dHandle;

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

    preferredDimSize_ = k.preferredDimSize_;

    return *this;
  }

  template <>
  kernel_t<COI>::~kernel_t(){}

  template <>
  std::string kernel_t<COI>::getCachedBinaryName(const std::string &filename,
                                                 kernelInfo &info_){

    std::string cachedBinary = getCachedName(filename,
                                             dHandle->getInfoSalt(info_));

    std::string libPath, soname;

    getFilePrefixAndName(cachedBinary, libPath, soname);

    std::string libName = "lib" + soname + ".so";

    return (libPath + libName);
  }

  template <>
  kernel_t<COI>* kernel_t<COI>::buildFromSource(const std::string &filename,
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

    std::string iCachedBinary = createIntermediateSource(filename,
                                                         cachedBinary,
                                                         info);

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

    const int compileError = system(sCommand.c_str());

    if(compileError){
      releaseFile(cachedBinary);
      OCCA_CHECK(false, "Compilation error");
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
                                                const std::string &functionName){
    OCCA_EXTRACT_DATA(COI, Kernel);

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

    OCCA_COI_CHECK("Kernel: Getting Handle",
                   COIProcessGetFunctionHandles(data_.chiefID,
                                                1,
                                                &c_functionName,
                                                &(data_.kernel)));

    return this;
  }

  template <>
  kernel_t<COI>* kernel_t<COI>::loadFromLibrary(const char *cache,
                                                const std::string &functionName){
    return buildFromBinary(cache, functionName);
  }

  // [-] Missing
  template <>
  int kernel_t<COI>::preferredDimSize(){
    preferredDimSize_ = 1;

    return preferredDimSize_;
  }

#include "operators/occaCOIKernelOperators.cpp"

  template <>
  void kernel_t<COI>::free(){
  }
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<COI>::memory_t(){
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
  memory_t<COI>::memory_t(const memory_t<COI> &m){
    *this = m;
  }

  template <>
  memory_t<COI>& memory_t<COI>::operator = (const memory_t<COI> &m){
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
  void memory_t<COI>::copyFrom(const void *src,
                               const uintptr_t bytes,
                               const uintptr_t offset){
    coiStream &stream = *((coiStream*) dHandle->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

    OCCA_COI_CHECK("Memory: Blocking on Memory Transfer",
                   COIEventWait(1, &(stream.lastEvent),
                                -1, true, NULL, NULL) );

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferWrite(*((coiMemory*) handle),
                                  offset,
                                  src,
                                  bytes_,
                                  COI_COPY_UNSPECIFIED,
                                  false, NULL,
                                  &(stream.lastEvent)));

    OCCA_COI_CHECK("Memory: Blocking on Memory Transfer",
                   COIEventWait(1, &(stream.lastEvent),
                                -1, true, NULL, NULL) );
  }

  template <>
  void memory_t<COI>::copyFrom(const memory_v *src,
                               const uintptr_t bytes,
                               const uintptr_t destOffset,
                               const uintptr_t srcOffset){
    coiStream &stream = *((coiStream*) dHandle->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + destOffset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << destOffset << " , " << (destOffset + bytes_) << " ]");

    OCCA_CHECK((bytes_ + srcOffset) <= src->size,
               "Source has size [" << src->size << "],"
               << "trying to access [ " << srcOffset << " , " << (srcOffset + bytes_) << " ]");

    OCCA_COI_CHECK("Memory: Blocking on Memory Transfer",
                   COIEventWait(1, &(stream.lastEvent),
                                -1, true, NULL, NULL) );

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferCopy(*((coiMemory*) handle),
                                 *((coiMemory*) src->handle),
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
    coiStream &stream = *((coiStream*) dHandle->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

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
    coiStream &stream = *((coiStream*) dHandle->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + srcOffset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << srcOffset << " , " << (srcOffset + bytes_) << " ]");

    OCCA_CHECK((bytes_ + destOffset) <= dest->size,
               "Destination has size [" << dest->size << "],"
               << "trying to access [ " << destOffset << " , " << (destOffset + bytes_) << " ]");

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
  void memory_t<COI>::asyncCopyFrom(const void *src,
                                    const uintptr_t bytes,
                                    const uintptr_t offset){
    coiStream &stream = *((coiStream*) dHandle->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

    OCCA_COI_CHECK("Memory: Blocking on Memory Transfer",
                   COIEventWait(1, &(stream.lastEvent),
                                -1, true, NULL, NULL) );

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferWrite(*((coiMemory*) handle),
                                  offset,
                                  src,
                                  bytes_,
                                  COI_COPY_UNSPECIFIED,
                                  false, NULL,
                                  &(stream.lastEvent)));
  }

  template <>
  void memory_t<COI>::asyncCopyFrom(const memory_v *src,
                                    const uintptr_t bytes,
                                    const uintptr_t destOffset,
                                    const uintptr_t srcOffset){
    coiStream &stream = *((coiStream*) dHandle->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + destOffset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << destOffset << " , " << (destOffset + bytes_) << " ]");

    OCCA_CHECK((bytes_ + srcOffset) <= src->size,
               "Source has size [" << src->size << "],"
               << "trying to access [ " << srcOffset << " , " << (srcOffset + bytes_) << " ]");

    OCCA_COI_CHECK("Memory: Blocking on Memory Transfer",
                   COIEventWait(1, &(stream.lastEvent),
                                -1, true, NULL, NULL) );

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferCopy(*((coiMemory*) handle),
                                 *((coiMemory*) src->handle),
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
    coiStream &stream = *((coiStream*) dHandle->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

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
    coiStream &stream = *((coiStream*) dHandle->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + srcOffset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << srcOffset << " , " << (srcOffset + bytes_) << " ]");

    OCCA_CHECK((bytes_ + destOffset) <= dest->size,
               "Destination has size [" << dest->size << "],"
               << "trying to access [ " << destOffset << " , " << (destOffset + bytes_) << " ]");

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
  void memory_t<COI>::mappedFree(){
    OCCA_COI_CHECK("Memory: free",
                   COIBufferDestroy( *((coiMemory*) handle) ) );

    delete handle;
    handle    = NULL;
    mappedPtr = NULL;

    size = 0;
  }

  template <>
  void memory_t<COI>::free(){
    OCCA_COI_CHECK("Memory: free",
                   COIBufferDestroy( *((coiMemory*) handle) ) );

    delete handle;
    handle = NULL;

    size = 0;
  }
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<COI>::device_t() {
    data = NULL;

    uvaEnabled_ = false;

    bytesAllocated = 0;

    getEnvironmentVariables();
  }

  template <>
  device_t<COI>::device_t(const device_t<COI> &d){
    data           = d.data;
    bytesAllocated = d.bytesAllocated;

    compilerFlags = d.compilerFlags;
  }

  template <>
  device_t<COI>& device_t<COI>::operator = (const device_t<COI> &d){
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
  void device_t<COI>::setup(argInfoMap &aim){
    data = new COIDeviceData_t;

    OCCA_EXTRACT_DATA(COI, Device);

    OCCA_CHECK(aim.has("deviceID"),
               "[COI] device not given [deviceID]");

    const int deviceID = aim.iGet("deviceID");

    uint32_t deviceCount;
    OCCA_COI_CHECK("Device: Get Count",
                   COIEngineGetCount(COI_ISA_MIC, &deviceCount));

    OCCA_CHECK(deviceID < deviceCount,
               "Trying to pick device [" << deviceID << "] out of the ["
               << deviceCount << "] COI devices available");

    OCCA_COI_CHECK("Device: Get Handle",
                   COIEngineGetHandle(COI_ISA_MIC, deviceID, &data_.deviceID) );

    coi::initDevice(data_);
  }

  template <>
  void device_t<COI>::addOccaHeadersToInfo(kernelInfo &info_){
    info_.addOCCAKeywords(occaCOIDefines);
  }

  template <>
  std::string device_t<COI>::getInfoSalt(const kernelInfo &info_){
    OCCA_EXTRACT_DATA(COI, Device);

    std::stringstream salt;

    salt << "COI"
         << info.salt()
         << parserVersion
         << compilerEnvScript
         << compiler
         << compilerFlags;

    return salt.str();
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
  void device_t<COI>::flush(){}

  template <>
  void device_t<COI>::finish(){
    coiStream &stream = *((coiStream*) currentStream);

    OCCA_COI_CHECK("Device: Waiting for Event",
                   COIEventWait(1, &(stream.lastEvent),
                                -1, true, NULL, NULL) );
  }

  template <>
  bool device_t<COI>::fakesUva(){
    return true;
  }

  template <>
  void device_t<COI>::waitFor(streamTag tag){
    finish(); // [-] Not done
  }

  template <>
  stream device_t<COI>::createStream(){
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

  template <>
  stream device_t<COI>::wrapStream(void *handle_){
    coiStream *retStream = new coiStream;
    retStream->handle = *((COIPIPELINE*) handle_);

    return retStream;
  }

  // [-] Event-based timing in COI?
  template <>
  streamTag device_t<COI>::tagStream(){
    streamTag ret;

    ret.tagTime = 0;

    return ret;
  }

  template <>
  double device_t<COI>::timeBetween(const streamTag &startTag, const streamTag &endTag){
    return (endTag.tagTime - startTag.tagTime);
  }

  template <>
  kernel_v* device_t<COI>::buildKernelFromSource(const std::string &filename,
                                                 const std::string &functionName,
                                                 const kernelInfo &info_){
    OCCA_EXTRACT_DATA(COI, Device);

    kernel_v *k = new kernel_t<COI>;

    k->dHandle = this;
    k->data    = new COIKernelData_t;

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

    k->dHandle = this;
    k->data    = new COIKernelData_t;

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
    kernel tmpK = occa::device(this).buildKernelFromSource(filename, functionName, info_);
    tmpK.free();

    kernelInfo info = info_;

    addOccaHeadersToInfo(info);

    std::string cachedBinary = getCachedName(filename, getInfoSalt(info));
    std::string libPath, soname;

    getFilePrefixAndName(cachedBinary, libPath, soname);

    std::string libName = "lib" + soname + ".so";

    cachedBinary = libPath + libName;
    //==================================

    library::infoID_t infoID;

    infoID.modelID    = modelID_;
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
                                                 const std::string &functionName){
    OCCA_EXTRACT_DATA(COI, Device);

    kernel_v *k = new kernel_t<COI>;

    k->dHandle = this;
    k->data    = new COIKernelData_t;

    COIKernelData_t &kData_ = *((COIKernelData_t*) k->data);

    kData_.chiefID = data_.chiefID;

    k->loadFromLibrary(cache, functionName);
    return k;
  }

  template <>
  memory_v* device_t<COI>::wrapMemory(void *handle_,
                                      const uintptr_t bytes){
    memory_v *mem = new memory_t<COI>;

    mem->dHandle = this;
    mem->size    = bytes;
    mem->handle  = handle_;

    mem->isAWrapper = true;

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
                                  void *src){
    OCCA_EXTRACT_DATA(COI, Device);

    memory_v *mem = new memory_t<COI>;

    mem->dHandle = this;
    mem->handle  = new coiMemory;
    mem->size    = bytes;

    OCCA_COI_CHECK("Device: Malloc",
                   COIBufferCreate(bytes,
                                   COI_BUFFER_NORMAL,
                                   0,
                                   src,
                                   1,
                                   &(data_.chiefID),
                                   (coiMemory*) mem->handle) );

    return mem;
  }

  template <>
  memory_v* device_t<COI>::textureAlloc(const int dim, const occa::dim &dims,
                                        void *src,
                                        occa::formatType type, const int permissions){
#warning "Textures not supported in COI yet"

    memory_v *mem = new memory_t<COI>;

    mem->dHandle = this;
    mem->size   = ((dim == 1) ? dims.x : (dims.x * dims.y)) * type.bytes();

    mem->isTexture = true;
    mem->textureInfo.dim  = dim;
    mem->textureInfo.dims = dims;

    mem->handle = cpu::malloc(mem->size);

    ::memcpy(mem->handle, src, mem->size);

    mem->textureInfo.arg = mem->handle;
    mem->handle = &(mem->textureInfo);

    return mem;
  }

  template <>
  memory_v* device_t<COI>::mappedAlloc(const uintptr_t bytes,
                                       void *src){
#warning "Mapped allocation is not supported in [COI] yet"
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
