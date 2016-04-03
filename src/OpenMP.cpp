#if OCCA_OPENMP_ENABLED

#include "occa/Serial.hpp"
#include "occa/OpenMP.hpp"

#include <omp.h>

namespace occa {
  //---[ Helper Functions ]-----------
  namespace omp {
    std::string notSupported = "N/A";

    std::string baseCompilerFlag(const int vendor_){
      if(vendor_ & (cpu::vendor::GNU |
                    cpu::vendor::LLVM)){

        return "-fopenmp";
      }
      else if(vendor_ & (cpu::vendor::Intel |
                         cpu::vendor::Pathscale)){

        return "-openmp";
      }
      else if(vendor_ & cpu::vendor::IBM){
        return "-qsmp";
      }
      else if(vendor_ & cpu::vendor::PGI){
        return "-mp";
      }
      else if(vendor_ & cpu::vendor::HP){
        return "+Oopenmp";
      }
      else if(vendor_ & cpu::vendor::VisualStudio){
        return "/openmp";
      }
      else if(vendor_ & cpu::vendor::Cray){
        return "";
      }

      return omp::notSupported;
    }

    std::string compilerFlag(const int vendor_,
                             const std::string &compiler){

#if (OCCA_OS & (LINUX_OS | OSX_OS))
      std::stringstream ss;
      std::string flag = omp::notSupported;

      const std::string safeCompiler = removeSlashes(compiler);
      const std::string &hash = safeCompiler;

      const std::string testFilename   = sys::getFilename("[occa]/testing/ompTest.cpp");
      const std::string binaryFilename = sys::getFilename("[occa]/testing/omp_" + safeCompiler);
      const std::string infoFilename   = sys::getFilename("[occa]/testing/ompInfo_" + safeCompiler);

      cacheFile(testFilename,
                readFile(env::OCCA_DIR + "/scripts/ompTest.cpp"),
                "ompTest");

      if(!haveHash(hash)){
        waitForHash(hash);
      } else {
        if(!sys::fileExists(infoFilename)){
          flag = baseCompilerFlag(vendor_);
          ss << compiler
             << ' '
             << flag
             << ' '
             << testFilename
             << " -o "
             << binaryFilename
             << " > /dev/null 2>&1";

          const int compileError = system(ss.str().c_str());

          if(compileError)
            flag = omp::notSupported;

          writeToFile(infoFilename, flag);
          releaseHash(hash);

          return flag;
        }
        releaseHash(hash);
      }

      ss << readFile(infoFilename);
      ss >> flag;

      return flag;
#elif (OCCA_OS == WINDOWS_OS)
      return "/openmp"; // VS Compilers support OpenMP
#endif
    }
  }
  //==================================


  //---[ Kernel ]---------------------
  template <>
  kernel_t<OpenMP>::kernel_t(){
    strMode = "OpenMP";

    data    = NULL;
    dHandle = NULL;

    dims  = 1;
    inner = occa::dim(1,1,1);
    outer = occa::dim(1,1,1);
  }

  template <>
  kernel_t<OpenMP>::kernel_t(const kernel_t<OpenMP> &k){
    *this = k;
  }

  template <>
  kernel_t<OpenMP>& kernel_t<OpenMP>::operator = (const kernel_t<OpenMP> &k){
    data    = k.data;
    dHandle = k.dHandle;

    metaInfo = k.metaInfo;

    dims  = k.dims;
    inner = k.inner;
    outer = k.outer;

    nestedKernels = k.nestedKernels;

    return *this;
  }

  template <>
  kernel_t<OpenMP>::~kernel_t(){}

  template <>
  void* kernel_t<OpenMP>::getKernelHandle(){
    OCCA_EXTRACT_DATA(OpenMP, Kernel);

    void *ret;

    ::memcpy(&ret, &data_.handle, sizeof(void*));

    return ret;
  }

  template <>
  void* kernel_t<OpenMP>::getProgramHandle(){
    OCCA_EXTRACT_DATA(OpenMP, Kernel);

    return data_.dlHandle;
  }

  template <>
  std::string kernel_t<OpenMP>::fixBinaryName(const std::string &filename){
#if (OCCA_OS & (LINUX_OS | OSX_OS))
    return filename;
#else
    return (filename + ".dll");
#endif
  }

  template <>
  kernel_t<OpenMP>* kernel_t<OpenMP>::buildFromSource(const std::string &filename,
                                                      const std::string &functionName,
                                                      const kernelInfo &info_){

    name = functionName;

    kernelInfo info = info_;

    dHandle->addOccaHeadersToInfo(info);

    const std::string hash = getFileContentHash(filename,
                                                dHandle->getInfoSalt(info));

    const std::string hashDir    = hashDirFor(filename, hash);
    const std::string sourceFile = hashDir + kc::sourceFile;
    const std::string binaryFile = hashDir + fixBinaryName(kc::binaryFile);
    bool foundBinary = true;

    if (!haveHash(hash, 0))
      waitForHash(hash, 0);
    else if (sys::fileExists(binaryFile))
      releaseHash(hash, 0);
    else
      foundBinary = false;

    if (foundBinary) {
      if(verboseCompilation_f)
        std::cout << "Found cached binary of [" << compressFilename(filename) << "] in [" << compressFilename(binaryFile) << "]\n";

      return buildFromBinary(binaryFile, functionName);
    }

    data = new OpenMPKernelData_t;

    createSourceFileFrom(filename, hashDir, info);

    std::stringstream command;

    if(dHandle->compilerEnvScript.size())
      command << dHandle->compilerEnvScript << " && ";

    //---[ Check if compiler flag is added ]------
    OpenMPDeviceData_t &dData_ = *((OpenMPDeviceData_t*) dHandle->data);

    const std::string ompFlag = dData_.OpenMPFlag;

    if((dHandle->compilerFlags.find(ompFlag) == std::string::npos) &&
       (            info.flags.find(ompFlag) == std::string::npos)){

      info.flags += ' ';
      info.flags += ompFlag;
    }
    //============================================

#if (OCCA_OS & (LINUX_OS | OSX_OS))
    command << dHandle->compiler
            << ' '    << dHandle->compilerFlags
            << ' '    << info.flags
            << ' '    << sourceFile
            << " -o " << binaryFile
            << " -I"  << env::OCCA_DIR << "/include"
            << " -L"  << env::OCCA_DIR << "/lib -locca"
            << std::endl;
#else
#  if (OCCA_DEBUG_ENABLED)
    std::string occaLib = env::OCCA_DIR + "\\lib\\libocca_d.lib ";
#  else
    std::string occaLib = env::OCCA_DIR + "\\lib\\libocca.lib ";
#  endif

    std::string ptLib   = env::OCCA_DIR + "\\lib\\pthreadVC2.lib ";

    command << dHandle->compiler
            << " /D MC_CL_EXE"
            << ' '    << dHandle->compilerFlags
            << ' '    << info.flags
            << " /I"  << env::OCCA_DIR << "\\include"
            << ' '    << sourceFile
            << " /link " << occaLib << ptLib << " /OUT:" << binaryFile
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
      releaseHash(hash, 0);
      OCCA_CHECK(false, "Compilation error");
    }

    OCCA_EXTRACT_DATA(OpenMP, Kernel);

    data_.dlHandle = cpu::dlopen(binaryFile, hash);
    data_.handle   = cpu::dlsym(data_.dlHandle, functionName, hash);

    releaseHash(hash, 0);

    return this;
  }

  template <>
  kernel_t<OpenMP>* kernel_t<OpenMP>::buildFromBinary(const std::string &filename,
                                                      const std::string &functionName){

    name = functionName;

    data = new OpenMPKernelData_t;

    OCCA_EXTRACT_DATA(OpenMP, Kernel);

    data_.dlHandle = cpu::dlopen(filename);
    data_.handle   = cpu::dlsym(data_.dlHandle, functionName);

    return this;
  }

  template <>
  kernel_t<OpenMP>* kernel_t<OpenMP>::loadFromLibrary(const char *cache,
                                                      const std::string &functionName){
    return buildFromBinary(cache, functionName);
  }

  template <>
  uintptr_t kernel_t<OpenMP>::maximumInnerDimSize(){
    return ((uintptr_t) -1);
  }

  // [-] Missing
  template <>
  int kernel_t<OpenMP>::preferredDimSize(){
    preferredDimSize_ = OCCA_SIMD_WIDTH;
    return OCCA_SIMD_WIDTH;
  }


  template <>
  void kernel_t<OpenMP>::runFromArguments(const int kArgc, const kernelArg *kArgs){
    OpenMPKernelData_t &data_ = *((OpenMPKernelData_t*) data);
    handleFunction_t tmpKernel = (handleFunction_t) data_.handle;
    int occaKernelArgs[6];

    occaKernelArgs[0] = outer.z; occaKernelArgs[3] = inner.z;
    occaKernelArgs[1] = outer.y; occaKernelArgs[4] = inner.y;
    occaKernelArgs[2] = outer.x; occaKernelArgs[5] = inner.x;

    int argc = 0;
    for(int i = 0; i < kArgc; ++i){
      for(int j = 0; j < kArgs[i].argc; ++j){
        data_.vArgs[argc++] = kArgs[i].args[j].ptr();
      }
    }

    int occaInnerId0 = 0, occaInnerId1 = 0, occaInnerId2 = 0;

    cpu::runFunction(tmpKernel,
                     occaKernelArgs,
                     occaInnerId0, occaInnerId1, occaInnerId2,
                     argc, data_.vArgs);
  }

  template <>
  void kernel_t<OpenMP>::free(){
    OCCA_EXTRACT_DATA(OpenMP, Kernel);

#if (OCCA_OS & (LINUX_OS | OSX_OS))
    dlclose(data_.dlHandle);
#else
    FreeLibrary((HMODULE) (data_.dlHandle));
#endif
  }
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<OpenMP>::memory_t(){
    strMode = "OpenMP";

    memInfo = memFlag::none;

    handle    = NULL;
    mappedPtr = NULL;
    uvaPtr    = NULL;

    dHandle = NULL;
    size    = 0;

    textureInfo.arg = NULL;
    textureInfo.dim = 1;
    textureInfo.w = textureInfo.h = textureInfo.d = 0;
  }

  template <>
  memory_t<OpenMP>::memory_t(const memory_t<OpenMP> &m){
    *this = m;
  }

  template <>
  memory_t<OpenMP>& memory_t<OpenMP>::operator = (const memory_t<OpenMP> &m){
    memInfo = m.memInfo;

    handle    = m.handle;
    mappedPtr = m.mappedPtr;
    uvaPtr    = m.uvaPtr;

    dHandle = m.dHandle;
    size    = m.size;

    textureInfo.arg  = m.textureInfo.arg;
    textureInfo.dim  = m.textureInfo.dim;

    textureInfo.w = m.textureInfo.w;
    textureInfo.h = m.textureInfo.h;
    textureInfo.d = m.textureInfo.d;

    if(isATexture())
      handle = &textureInfo;

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
  void memory_t<OpenMP>::copyFrom(const void *src,
                                  const uintptr_t bytes,
                                  const uintptr_t offset){
    dHandle->finish();

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

    void *destPtr      = ((char*) (isATexture() ? textureInfo.arg : handle)) + offset;
    const void *srcPtr = src;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<OpenMP>::copyFrom(const memory_v *src,
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

    void *destPtr      = ((char*) (isATexture()      ? textureInfo.arg      : handle))      + destOffset;
    const void *srcPtr = ((char*) (src->isATexture() ? src->textureInfo.arg : src->handle)) + srcOffset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<OpenMP>::copyTo(void *dest,
                                const uintptr_t bytes,
                                const uintptr_t offset){
    dHandle->finish();

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

    void *destPtr      = dest;
    const void *srcPtr = ((char*) (isATexture() ? textureInfo.arg : handle)) + offset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<OpenMP>::copyTo(memory_v *dest,
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

    void *destPtr      = ((char*) (dest->isATexture() ? dest->textureInfo.arg : dest->handle)) + destOffset;
    const void *srcPtr = ((char*) (isATexture() ? textureInfo.arg : handle))       + srcOffset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<OpenMP>::asyncCopyFrom(const void *src,
                                       const uintptr_t bytes,
                                       const uintptr_t offset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

    void *destPtr      = ((char*) (isATexture() ? textureInfo.arg : handle)) + offset;
    const void *srcPtr = src;


    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<OpenMP>::asyncCopyFrom(const memory_v *src,
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

    void *destPtr      = ((char*) (isATexture()      ? textureInfo.arg      : handle))      + destOffset;
    const void *srcPtr = ((char*) (src->isATexture() ? src->textureInfo.arg : src->handle)) + srcOffset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<OpenMP>::asyncCopyTo(void *dest,
                                     const uintptr_t bytes,
                                     const uintptr_t offset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

    void *destPtr      = dest;
    const void *srcPtr = ((char*) (isATexture() ? textureInfo.arg : handle)) + offset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<OpenMP>::asyncCopyTo(memory_v *dest,
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

    void *destPtr      = ((char*) (dest->isATexture() ? dest->textureInfo.arg : dest->handle)) + destOffset;
    const void *srcPtr = ((char*) (isATexture() ? textureInfo.arg : handle))       + srcOffset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<OpenMP>::mappedFree(){
    cpu::free(handle);
    handle    = NULL;
    mappedPtr = NULL;

    size = 0;
  }

  template <>
  void memory_t<OpenMP>::free(){
    if(isATexture()){
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
  device_t<OpenMP>::device_t(){
    strMode = "OpenMP";

    data = NULL;

    uvaEnabled_ = false;

    bytesAllocated = 0;

    getEnvironmentVariables();

    cpu::addSharedBinaryFlagsTo(compiler, compilerFlags);
  }

  template <>
  device_t<OpenMP>::device_t(const device_t<OpenMP> &d){
    *this = d;
  }

  template <>
  device_t<OpenMP>& device_t<OpenMP>::operator = (const device_t<OpenMP> &d){
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
  void* device_t<OpenMP>::getContextHandle(){
    return NULL;
  }

  template <>
  void device_t<OpenMP>::setup(argInfoMap &aim){
    properties = aim;

    // Generate an OpenMP library dependency (so it doesn't crash when dlclose())
    omp_get_num_threads();

    data = new OpenMPDeviceData_t;

    OCCA_EXTRACT_DATA(OpenMP, Device);

    data_.vendor         = cpu::compilerVendor(compiler);
    data_.OpenMPFlag     = omp::compilerFlag(data_.vendor, compiler);
    data_.supportsOpenMP = (data_.OpenMPFlag != omp::notSupported);

    cpu::addSharedBinaryFlagsTo(data_.vendor, compilerFlags);
  }

  template <>
  void device_t<OpenMP>::addOccaHeadersToInfo(kernelInfo &info_){
    info_.mode = OpenMP;
  }

  template <>
  std::string device_t<OpenMP>::getInfoSalt(const kernelInfo &info_){
    std::stringstream salt;

    salt << "OpenMP"
         << info_.salt()
         << parserVersion
         << compilerEnvScript
         << compiler
         << compilerFlags;

    return salt.str();
  }

  template <>
  deviceIdentifier device_t<OpenMP>::getIdentifier() const {
    deviceIdentifier dID;

    dID.mode_ = OpenMP;

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
  void device_t<OpenMP>::getEnvironmentVariables(){
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
    compilerFlags = " /Od /openmp";
#  else
    compilerFlags = " /O2 /openmp";
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
  void device_t<OpenMP>::appendAvailableDevices(std::vector<device> &dList){
    device d;
    d.setup("OpenMP");

    dList.push_back(d);
  }

  template <>
  void device_t<OpenMP>::setCompiler(const std::string &compiler_){
    compiler = compiler_;

    OCCA_EXTRACT_DATA(OpenMP, Device);

    data_.vendor         = cpu::compilerVendor(compiler);
    data_.OpenMPFlag     = omp::compilerFlag(data_.vendor, compiler);
    data_.supportsOpenMP = (data_.OpenMPFlag != omp::notSupported);

    cpu::addSharedBinaryFlagsTo(data_.vendor, compilerFlags);
  }

  template <>
  void device_t<OpenMP>::setCompilerEnvScript(const std::string &compilerEnvScript_){
    compilerEnvScript = compilerEnvScript_;
  }

  template <>
  void device_t<OpenMP>::setCompilerFlags(const std::string &compilerFlags_){
    OCCA_EXTRACT_DATA(OpenMP, Device);

    compilerFlags = compilerFlags_;

    cpu::addSharedBinaryFlagsTo(data_.vendor, compilerFlags);
  }

  template <>
  void device_t<OpenMP>::flush(){}

  template <>
  void device_t<OpenMP>::finish(){}

  template <>
  bool device_t<OpenMP>::fakesUva(){
    return false;
  }

  template <>
  void device_t<OpenMP>::waitFor(streamTag tag){}

  template <>
  stream_t device_t<OpenMP>::createStream(){
    return NULL;
  }

  template <>
  void device_t<OpenMP>::freeStream(stream_t s){}

  template <>
  stream_t device_t<OpenMP>::wrapStream(void *handle_){
    return NULL;
  }

  template <>
  streamTag device_t<OpenMP>::tagStream(){
    streamTag ret;

    ret.tagTime = currentTime();

    return ret;
  }

  template <>
  double device_t<OpenMP>::timeBetween(const streamTag &startTag, const streamTag &endTag){
    return (endTag.tagTime - startTag.tagTime);
  }

  template <>
  std::string device_t<OpenMP>::fixBinaryName(const std::string &filename){
#if (OCCA_OS & (LINUX_OS | OSX_OS))
    return filename;
#else
    return (filename + ".dll");
#endif
  }

  template <>
  kernel_v* device_t<OpenMP>::buildKernelFromSource(const std::string &filename,
                                                    const std::string &functionName,
                                                    const kernelInfo &info_){
    OCCA_EXTRACT_DATA(OpenMP, Device);

    kernel_v *k;

    if(data_.supportsOpenMP){
      k = new kernel_t<OpenMP>;
    }
    else{
      std::cout << "Compiler [" << compiler << "] does not support OpenMP, defaulting to [Serial] mode\n";
      k = new kernel_t<Serial>;
    }

    k->dHandle = this;

    k->buildFromSource(filename, functionName, info_);

    return k;
  }

  template <>
  kernel_v* device_t<OpenMP>::buildKernelFromBinary(const std::string &filename,
                                                    const std::string &functionName){
    OCCA_EXTRACT_DATA(OpenMP, Device);

    kernel_v *k;

    if(data_.supportsOpenMP){
      k = new kernel_t<OpenMP>;
    }
    else{
      std::cout << "Compiler [" << compiler << "] does not support OpenMP, defaulting to [Serial] mode\n";
      k = new kernel_t<Serial>;
    }

    k->dHandle = this;

    k->buildFromBinary(filename, functionName);

    return k;
  }

  template <>
  void device_t<OpenMP>::cacheKernelInLibrary(const std::string &filename,
                                              const std::string &functionName,
                                              const kernelInfo &info_){
#if 0
    //---[ Creating shared library ]----
    kernel tmpK = occa::device(this).buildKernelFromSource(filename, functionName, info_);
    tmpK.free();

    kernelInfo info = info_;

    addOccaHeadersToInfo(info);

    std::string cachedBinary = getCachedName(filename, getInfoSalt(info));

#if (OCCA_OS == WINDOWS_OS)
    // Windows requires .dll extension
    cachedBinary = cachedBinary + ".dll";
#endif
    //==================================

    library::infoID_t infoID;

    infoID.modelID    = modelID_;
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
#endif
  }

  template <>
  kernel_v* device_t<OpenMP>::loadKernelFromLibrary(const char *cache,
                                                    const std::string &functionName){
#if 0
    kernel_v *k = new kernel_t<OpenMP>;
    k->dHandle = this;
    k->loadFromLibrary(cache, functionName);
    return k;
#endif
    return NULL;
  }

  template <>
  memory_v* device_t<OpenMP>::wrapMemory(void *handle_,
                                         const uintptr_t bytes){
    memory_v *mem = new memory_t<OpenMP>;

    mem->dHandle = this;
    mem->size    = bytes;
    mem->handle  = handle_;

    mem->memInfo |= memFlag::isAWrapper;

    return mem;
  }

  template <>
  memory_v* device_t<OpenMP>::wrapTexture(void *handle_,
                                          const int dim, const occa::dim &dims,
                                          occa::formatType type, const int permissions){
    memory_v *mem = new memory_t<OpenMP>;

    mem->dHandle = this;
    mem->size    = ((dim == 1) ? dims.x : (dims.x * dims.y)) * type.bytes();

    mem->memInfo |= (memFlag::isATexture |
                     memFlag::isAWrapper);

    mem->textureInfo.dim  = dim;

    mem->textureInfo.w = dims.x;
    mem->textureInfo.h = dims.y;
    mem->textureInfo.d = dims.z;

    mem->textureInfo.arg = handle_;

    mem->handle = &(mem->textureInfo);

    return mem;
  }

  template <>
  memory_v* device_t<OpenMP>::malloc(const uintptr_t bytes,
                                     void *src){
    memory_v *mem = new memory_t<OpenMP>;

    mem->dHandle = this;
    mem->size    = bytes;

    mem->handle = cpu::malloc(bytes);

    if(src != NULL)
      ::memcpy(mem->handle, src, bytes);

    return mem;
  }

  template <>
  memory_v* device_t<OpenMP>::textureAlloc(const int dim, const occa::dim &dims,
                                           void *src,
                                           occa::formatType type, const int permissions){
    memory_v *mem = new memory_t<OpenMP>;

    mem->dHandle = this;
    mem->size    = ((dim == 1) ? dims.x : (dims.x * dims.y)) * type.bytes();

    mem->memInfo |= memFlag::isATexture;

    mem->textureInfo.dim  = dim;

    mem->textureInfo.w = dims.x;
    mem->textureInfo.h = dims.y;
    mem->textureInfo.d = dims.z;

    mem->handle = cpu::malloc(mem->size);

    ::memcpy(mem->textureInfo.arg, src, mem->size);

    mem->handle = &(mem->textureInfo);

    return mem;
  }

  template <>
  memory_v* device_t<OpenMP>::mappedAlloc(const uintptr_t bytes,
                                          void *src){
    memory_v *mem = malloc(bytes, src);

    mem->mappedPtr = mem->handle;

    return mem;
  }

  template <>
  uintptr_t device_t<OpenMP>::memorySize(){
    return cpu::installedRAM();
  }

  template <>
  void device_t<OpenMP>::free(){}

  template <>
  int device_t<OpenMP>::simdWidth(){
    simdWidth_ = OCCA_SIMD_WIDTH;
    return OCCA_SIMD_WIDTH;
  }
  //==================================
}

#endif
