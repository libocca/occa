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
         << dev->dHandle->compilerEnvScript
         << dev->dHandle->compiler
         << dev->dHandle->compilerFlags
         << functionName;

	struct stat buffer;
    std::string cachedBinary = getCachedName(filename, salt.str());

#ifdef WIN32
	cachedBinary = cachedBinary + ".dll"; // windows refuses to load dll's that do not end with '.dll'
#endif

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

    OCCA_EXTRACT_DATA(OpenMP, Kernel);

#ifndef WIN32
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


#ifndef WIN32
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

#ifndef WIN32
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



#ifndef WIN32
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
  void kernel_t<OpenMP>::free(){
    // [-] Fix later
    OCCA_EXTRACT_DATA(OpenMP, Kernel);
#ifndef WIN32
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
  }

  template <>
  memory_t<OpenMP>::memory_t(const memory_t<OpenMP> &m){
    handle = m.handle;
    dev    = m.dev;
    size   = m.size;
  }

  template <>
  memory_t<OpenMP>& memory_t<OpenMP>::operator = (const memory_t<OpenMP> &m){
    handle = m.handle;
    dev    = m.dev;
    size   = m.size;

    return *this;
  }

  template <>
  memory_t<OpenMP>::~memory_t(){}

  template <>
  void memory_t<OpenMP>::copyFrom(const void *source,
                                  const uintptr_t bytes,
                                  const uintptr_t offset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    ::memcpy(((char*) handle) + offset, source, bytes_);
  }

  template <>
  void memory_t<OpenMP>::copyFrom(const memory_v *source,
                                  const uintptr_t bytes,
                                  const uintptr_t destOffset,
                                  const uintptr_t srcOffset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + destOffset) <= size);
    OCCA_CHECK((bytes_ + srcOffset)  <= source->size);

    ::memcpy(((char*) handle)         + destOffset,
             ((char*) source->handle) + srcOffset,
             bytes_);
  }

  template <>
  void memory_t<OpenMP>::copyTo(void *dest,
                                const uintptr_t bytes,
                                const uintptr_t offset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    ::memcpy(dest, ((char*) handle) + offset, bytes_);
  }

  template <>
  void memory_t<OpenMP>::copyTo(memory_v *dest,
                                const uintptr_t bytes,
                                const uintptr_t destOffset,
                                const uintptr_t srcOffset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + srcOffset)  <= size);
    OCCA_CHECK((bytes_ + destOffset) <= dest->size);

    ::memcpy(((char*) dest->handle) + destOffset,
             ((char*) handle)       + srcOffset,
             bytes_);
  }

  template <>
  void memory_t<OpenMP>::asyncCopyFrom(const void *source,
                                       const uintptr_t bytes,
                                       const uintptr_t offset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    ::memcpy(((char*) handle) + offset, source , bytes_);
  }

  template <>
  void memory_t<OpenMP>::asyncCopyFrom(const memory_v *source,
                                       const uintptr_t bytes,
                                       const uintptr_t destOffset,
                                       const uintptr_t srcOffset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + destOffset) <= size);
    OCCA_CHECK((bytes_ + srcOffset)  <= source->size);

    ::memcpy(((char*) handle)         + destOffset,
             ((char*) source->handle) + srcOffset,
             bytes_);
  }

  template <>
  void memory_t<OpenMP>::asyncCopyTo(void *dest,
                                     const uintptr_t bytes,
                                     const uintptr_t offset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    ::memcpy(dest, ((char*) handle) + offset, bytes_);
  }

  template <>
  void memory_t<OpenMP>::asyncCopyTo(memory_v *dest,
                                     const uintptr_t bytes,
                                     const uintptr_t destOffset,
                                     const uintptr_t srcOffset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + srcOffset)  <= size);
    OCCA_CHECK((bytes_ + destOffset) <= dest->size);

    ::memcpy(((char*) dest->handle) + destOffset,
             ((char*) handle)       + srcOffset,
             bytes_);
  }

  template <>
  void memory_t<OpenMP>::free(){
    delete (char*) handle;
    size = 0;
  }
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<OpenMP>::device_t(){
    data = NULL;
    memoryUsed = 0;

    getEnvironmentVariables();
  }

  template <>
  device_t<OpenMP>::device_t(int platform, int device){
    data       = NULL;
    memoryUsed = 0;

    getEnvironmentVariables();
  }

  template <>
  device_t<OpenMP>::device_t(const device_t<OpenMP> &d){
    data       = d.data;
    memoryUsed = d.memoryUsed;

    compiler      = d.compiler;
    compilerFlags = d.compilerFlags;
  }

  template <>
  device_t<OpenMP>& device_t<OpenMP>::operator = (const device_t<OpenMP> &d){
    data       = d.data;
    memoryUsed = d.memoryUsed;

    compiler      = d.compiler;
    compilerFlags = d.compilerFlags;

    return *this;
  }

  template <>
  void device_t<OpenMP>::setup(const int unusedArg1, const int unusedArg2){}

  template <>
  void device_t<OpenMP>::getEnvironmentVariables(){
    char *c_compiler = getenv("OCCA_OPENMP_COMPILER");

    if(c_compiler != NULL)
      compiler = std::string(c_compiler);
    else
      compiler = "g++";

    char *c_compilerFlags = getenv("OCCA_OPENMP_COMPILER_FLAGS");

    if(c_compilerFlags != NULL)
      compilerFlags = std::string(c_compilerFlags);
    else{
#if OCCA_DEBUG_ENABLED
      compilerFlags = "-g";
#else
      compilerFlags = "-D__extern_always_inline=inline -O3";
#endif
    }
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
  void device_t<OpenMP>::free(){}

  template <>
  int device_t<OpenMP>::simdWidth(){
    simdWidth_ = OCCA_SIMD_WIDTH;
    return OCCA_SIMD_WIDTH;
  }
  //==================================
};
