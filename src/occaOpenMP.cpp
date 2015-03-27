#if OCCA_OPENMP_ENABLED

#include "occaSerial.hpp"
#include "occaOpenMP.hpp"

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
      const std::string safeCompiler = removeSlashes(compiler);
      const std::string cacheDir     = getCachePath();

      const std::string testFilename = (cacheDir + "/.ompTest.cpp");
      const std::string infoFilename = (cacheDir + std::string("/.ompTest_") + safeCompiler);

      if(!haveFile(testFilename)){
        waitForFile(testFilename);
      }
      else{
        if(!fileExists(testFilename)){
          std::string testContent = ("#include <omp.h>\n"
                                     "\n"
                                     "int main(int argc, char **argv){\n"
                                     "#ifdef _OPENMP\n"
                                     "  return 0;\n"
                                     "#else\n"
                                     "  return 1;\n"
                                     "#endif\n"
                                     "}");

          writeToFile(testFilename, testContent);
        }

        std::string binaryFilename = (getCachePath() + std::string("/.ompBinary_" + safeCompiler));

        if(!fileExists(infoFilename)){
          std::string flag = baseCompilerFlag(vendor_);
          int compileError = 1;

          std::string sCommand;

          if(flag != omp::notSupported){
            sCommand += safeCompiler;
            sCommand += ' ';
            sCommand += flag;
            sCommand += ' ';
            sCommand += testFilename;
            sCommand += " -o ";
            sCommand += binaryFilename;
            sCommand += " > /dev/null 2>&1";

            compileError = system(sCommand.c_str());
          }

          if(compileError)
            flag = omp::notSupported;

          writeToFile(infoFilename, flag);

          releaseFile(testFilename);

          return flag;
        }
        else
          releaseFile(testFilename);
      }

      return readFile(infoFilename);

#elif (OCCA_OS == WINDOWS_OS)
      return "/openmp"; // VS Compilers support OpenMP
#endif
    }
  };
  //==================================


  //---[ Kernel ]---------------------
  template <>
  kernel_t<OpenMP>::kernel_t(){
    data    = NULL;
    dHandle = NULL;

    dims  = 1;
    inner = occa::dim(1,1,1);
    outer = occa::dim(1,1,1);

    nestedKernelCount = 0;
    nestedKernels     = NULL;
  }

  template <>
  kernel_t<OpenMP>::kernel_t(const kernel_t<OpenMP> &k){
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
  }

  template <>
  kernel_t<OpenMP>& kernel_t<OpenMP>::operator = (const kernel_t<OpenMP> &k){
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

    return *this;
  }

  template <>
  kernel_t<OpenMP>::~kernel_t(){}

  template <>
  std::string kernel_t<OpenMP>::getCachedBinaryName(const std::string &filename,
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
  kernel_t<OpenMP>* kernel_t<OpenMP>::buildFromSource(const std::string &filename,
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

    data = new OpenMPKernelData_t;

    std::string iCachedBinary = createIntermediateSource(filename,
                                                         cachedBinary,
                                                         info);

    const std::string occaDir = getOCCADir();

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

    OCCA_EXTRACT_DATA(OpenMP, Kernel);

    data_.dlHandle = cpu::dlopen(cachedBinary, true);
    data_.handle   = cpu::dlsym(data_.dlHandle, cachedBinary, functionName, true);

    releaseFile(cachedBinary);

    return this;
  }

  template <>
  kernel_t<OpenMP>* kernel_t<OpenMP>::buildFromBinary(const std::string &filename,
                                                      const std::string &functionName){
    data = new OpenMPKernelData_t;

    OCCA_EXTRACT_DATA(OpenMP, Kernel);

    data_.dlHandle = cpu::dlopen(filename, false);
    data_.handle   = cpu::dlsym(data_.dlHandle, filename, functionName, false);

    return this;
  }

  template <>
  kernel_t<OpenMP>* kernel_t<OpenMP>::loadFromLibrary(const char *cache,
                                                      const std::string &functionName){
    return buildFromBinary(cache, functionName);
  }

  // [-] Missing
  template <>
  int kernel_t<OpenMP>::preferredDimSize(){
    preferredDimSize_ = OCCA_SIMD_WIDTH;
    return OCCA_SIMD_WIDTH;
  }

#include "operators/occaOpenMPKernelOperators.cpp"

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
  memory_t<OpenMP>::memory_t(const memory_t<OpenMP> &m){
    *this = m;
  }

  template <>
  memory_t<OpenMP>& memory_t<OpenMP>::operator = (const memory_t<OpenMP> &m){
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

    void *destPtr      = ((char*) (isTexture ? textureInfo.arg : handle)) + offset;
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

    void *destPtr      = ((char*) (isTexture      ? textureInfo.arg      : handle))      + destOffset;
    const void *srcPtr = ((char*) (src->isTexture ? src->textureInfo.arg : src->handle)) + srcOffset;

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
    const void *srcPtr = ((char*) (isTexture ? textureInfo.arg : handle)) + offset;

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

    void *destPtr      = ((char*) (dest->isTexture ? dest->textureInfo.arg : dest->handle)) + destOffset;
    const void *srcPtr = ((char*) (isTexture ? textureInfo.arg : handle))       + srcOffset;

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

    void *destPtr      = ((char*) (isTexture ? textureInfo.arg : handle)) + offset;
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

    void *destPtr      = ((char*) (isTexture      ? textureInfo.arg      : handle))      + destOffset;
    const void *srcPtr = ((char*) (src->isTexture ? src->textureInfo.arg : src->handle)) + srcOffset;

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
    const void *srcPtr = ((char*) (isTexture ? textureInfo.arg : handle)) + offset;

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

    void *destPtr      = ((char*) (dest->isTexture ? dest->textureInfo.arg : dest->handle)) + destOffset;
    const void *srcPtr = ((char*) (isTexture ? textureInfo.arg : handle))       + srcOffset;

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
  device_t<OpenMP>::device_t(){
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
  void device_t<OpenMP>::setup(argInfoMap &aim){
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
    OCCA_EXTRACT_DATA(OpenMP, Device);

    if(data_.supportsOpenMP)
      info_.addOCCAKeywords(occaOpenMPDefines);
    else
      info_.addOCCAKeywords(occaSerialDefines);
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
  stream device_t<OpenMP>::createStream(){
    return NULL;
  }

  template <>
  void device_t<OpenMP>::freeStream(stream s){}

  template <>
  stream device_t<OpenMP>::wrapStream(void *handle_){
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
  }

  template <>
  kernel_v* device_t<OpenMP>::loadKernelFromLibrary(const char *cache,
                                                    const std::string &functionName){
    kernel_v *k = new kernel_t<OpenMP>;
    k->dHandle = this;
    k->loadFromLibrary(cache, functionName);
    return k;
  }

  template <>
  memory_v* device_t<OpenMP>::wrapMemory(void *handle_,
                                         const uintptr_t bytes){
    memory_v *mem = new memory_t<OpenMP>;

    mem->dHandle = this;
    mem->size    = bytes;
    mem->handle  = handle_;

    mem->isAWrapper = true;

    return mem;
  }

  template <>
  memory_v* device_t<OpenMP>::wrapTexture(void *handle_,
                                          const int dim, const occa::dim &dims,
                                          occa::formatType type, const int permissions){
    memory_v *mem = new memory_t<OpenMP>;

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

    mem->isTexture = true;
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
  void device_t<OpenMP>::free(){}

  template <>
  int device_t<OpenMP>::simdWidth(){
    simdWidth_ = OCCA_SIMD_WIDTH;
    return OCCA_SIMD_WIDTH;
  }
  //==================================
};

#endif
