#include "occaPthreads.hpp"

namespace occa {
  //---[ Kernel ]---------------------
  template <>
  kernel_t<Pthreads>::kernel_t(){
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
  kernel_t<Pthreads>::kernel_t(const kernel_t<Pthreads> &k){
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
  kernel_t<Pthreads>& kernel_t<Pthreads>::operator = (const kernel_t<Pthreads> &k){
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
  kernel_t<Pthreads>::~kernel_t(){}

  template <>
  kernel_t<Pthreads>* kernel_t<Pthreads>::buildFromSource(const std::string &filename,
                                                          const std::string &functionName_,
                                                          const kernelInfo &info_){
    functionName = functionName_;

    kernelInfo info = info_;
    info.addDefine("OCCA_USING_CPU"     , 1);
    info.addDefine("OCCA_USING_PTHREADS", 1);

    info.addOCCAKeywords(occaPthreadsDefines);

    std::stringstream salt;
    salt << "Pthreads"
         << info.salt()
         << functionName;

    std::string cachedBinary = binaryIsCached(filename, salt.str());

    struct stat buffer;
    bool fileExists = (stat(cachedBinary.c_str(), &buffer) == 0);

    if(fileExists){
      std::cout << "Found cached binary of [" << filename << "] in [" << cachedBinary << "]\n";
      return buildFromBinary(cachedBinary, functionName);
    }

    data = new PthreadsKernelData_t;

    std::string iCachedBinary = createIntermediateSource(filename,
                                                         cachedBinary,
                                                         info);

    std::stringstream command;

    command << dev->dHandle->compiler
            << " -o " << cachedBinary
            << " -x c++ -w -fPIC -shared"
            << ' '    << dev->dHandle->compilerFlags
            << ' '    << info.flags
            << ' '    << iCachedBinary;

    const std::string &sCommand = command.str();

    std::cout << sCommand << '\n';

    system(sCommand.c_str());

    OCCA_EXTRACT_DATA(Pthreads, Kernel);

    data_.dlHandle = dlopen(cachedBinary.c_str(), RTLD_NOW);

    OCCA_CHECK(data_.dlHandle != NULL);

    data_.handle = dlsym(data_.dlHandle, functionName.c_str());

    char *dlError;
    if ((dlError = dlerror()) != NULL)  {
      fputs(dlError, stderr);
      throw 1;
    }

    PthreadsDeviceData_t &dData = *((PthreadsDeviceData_t*) ((device_t<Pthreads>*) dev->dHandle)->data);

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
  kernel_t<Pthreads>* kernel_t<Pthreads>::buildFromBinary(const std::string &filename,
                                                          const std::string &functionName_){
    data = new PthreadsKernelData_t;

    OCCA_EXTRACT_DATA(Pthreads, Kernel);

    functionName = functionName_;

    data_.dlHandle = dlopen(filename.c_str(), RTLD_LAZY | RTLD_LOCAL);

    OCCA_CHECK(data_.dlHandle != NULL);

    data_.handle = dlsym(data_.dlHandle, functionName.c_str());

    char *dlError;
    if ((dlError = dlerror()) != NULL)  {
      fputs(dlError, stderr);
      throw 1;
    }

    PthreadsDeviceData_t &dData = *((PthreadsDeviceData_t*) ((device_t<Pthreads>*) dev->dHandle)->data);

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

  // [-] Missing
  template <>
  int kernel_t<Pthreads>::preferredDimSize(){
    preferredDimSize_ = OCCA_SIMD_WIDTH;
    return OCCA_SIMD_WIDTH;
  }

  OCCA_PTHREADS_KERNEL_OPERATOR_DEFINITIONS;

  template <>
  double kernel_t<Pthreads>::timeTaken(){
    const double &start = *((double*) startTime);
    const double &end   = *((double*) endTime);

    return 1.0e3*(end - start);
  }

  template <>
  void kernel_t<Pthreads>::free(){
    // [-] Fix later
    OCCA_EXTRACT_DATA(Pthreads, Kernel);

    dlclose(data_.dlHandle);
  }
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<Pthreads>::memory_t(){
    handle = NULL;
    dev    = NULL;
    size = 0;
  }

  template <>
  memory_t<Pthreads>::memory_t(const memory_t<Pthreads> &m){
    handle = m.handle;
    dev    = m.dev;
    size   = m.size;
  }

  template <>
  memory_t<Pthreads>& memory_t<Pthreads>::operator = (const memory_t<Pthreads> &m){
    handle = m.handle;
    dev    = m.dev;
    size   = m.size;

    return *this;
  }

  template <>
  memory_t<Pthreads>::~memory_t(){}

  template <>
  void memory_t<Pthreads>::copyFrom(const void *source,
                                    const size_t bytes,
                                    const size_t offset){
    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    dev->finish();

    ::memcpy(((char*) handle) + offset, source, bytes_);
  }

  template <>
  void memory_t<Pthreads>::copyFrom(const memory_v *source,
                                    const size_t bytes,
                                    const size_t destOffset,
                                    const size_t srcOffset){
    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + destOffset) <= size);
    OCCA_CHECK((bytes_ + srcOffset)  <= source->size);

    dev->finish();

    ::memcpy(((char*) handle)         + destOffset,
             ((char*) source->handle) + srcOffset,
             bytes_);
  }

  template <>
  void memory_t<Pthreads>::copyTo(void *dest,
                                  const size_t bytes,
                                  const size_t offset){
    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    dev->finish();

    ::memcpy(dest, ((char*) handle) + offset, bytes_);
  }

  template <>
  void memory_t<Pthreads>::copyTo(memory_v *dest,
                                  const size_t bytes,
                                  const size_t destOffset,
                                  const size_t srcOffset){
    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + srcOffset)  <= size);
    OCCA_CHECK((bytes_ + destOffset) <= dest->size);

    dev->finish();

    ::memcpy(((char*) dest->handle) + destOffset,
             ((char*) handle)       + srcOffset,
             bytes_);
  }

  template <>
  void memory_t<Pthreads>::asyncCopyFrom(const void *source,
                                         const size_t bytes,
                                         const size_t offset){
    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    ::memcpy(((char*) handle) + offset, source , bytes_);
  }

  template <>
  void memory_t<Pthreads>::asyncCopyFrom(const memory_v *source,
                                         const size_t bytes,
                                         const size_t destOffset,
                                         const size_t srcOffset){
    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + destOffset) <= size);
    OCCA_CHECK((bytes_ + srcOffset)  <= source->size);

    ::memcpy(((char*) handle)         + destOffset,
             ((char*) source->handle) + srcOffset,
             bytes_);
  }

  template <>
  void memory_t<Pthreads>::asyncCopyTo(void *dest,
                                       const size_t bytes,
                                       const size_t offset){
    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    ::memcpy(dest, ((char*) handle) + offset, bytes_);
  }

  template <>
  void memory_t<Pthreads>::asyncCopyTo(memory_v *dest,
                                       const size_t bytes,
                                       const size_t destOffset,
                                       const size_t srcOffset){
    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + srcOffset)  <= size);
    OCCA_CHECK((bytes_ + destOffset) <= dest->size);

    ::memcpy(((char*) dest->handle) + destOffset,
             ((char*) handle)       + srcOffset,
             bytes_);
  }

  template <>
  void memory_t<Pthreads>::free(){
    ::free(handle);
  }
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<Pthreads>::device_t(){
    data = NULL;
    memoryUsed = 0;

    getEnvironmentVariables();
  }

  template <>
  device_t<Pthreads>::device_t(int platform, int device){
    data       = NULL;
    memoryUsed = 0;

    getEnvironmentVariables();
  }

  template <>
  device_t<Pthreads>::device_t(const device_t<Pthreads> &d){
    data       = d.data;
    memoryUsed = d.memoryUsed;

    compiler      = d.compiler;
    compilerFlags = d.compilerFlags;
  }

  template <>
  device_t<Pthreads>& device_t<Pthreads>::operator = (const device_t<Pthreads> &d){
    data       = d.data;
    memoryUsed = d.memoryUsed;

    compiler      = d.compiler;
    compilerFlags = d.compilerFlags;

    return *this;
  }

  template <>
  void device_t<Pthreads>::setup(const int threadCount, const int pinningInfo){
    data = new PthreadsDeviceData_t;

    OCCA_EXTRACT_DATA(Pthreads, Device);

    data_.pendingJobs = 0;

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
    data_.coreCount = sysconf(_SC_NPROCESSORS_ONLN);
#else
#  warning "Core finding not implemented for this OS"
#endif

    data_.pThreadCount = (threadCount ? threadCount : 1);
    data_.pinningInfo  = pinningInfo;

    int error = pthread_mutex_init(&(data_.pendingJobsMutex), NULL);
    OCCA_CHECK(error == 0);

    error = pthread_mutex_init(&(data_.kernelMutex), NULL);
    OCCA_CHECK(error == 0);

    for(int p = 0; p < data_.pThreadCount; ++p){
      PthreadWorkerData_t *args = new PthreadWorkerData_t;

      args->rank  = p;
      args->count = data_.pThreadCount;

      // [-] Need to know number of sockets
      if(pinningInfo & occa::compact)
        args->pinnedCore = p % data_.coreCount;
      else
        args->pinnedCore = p % data_.coreCount;

      args->pendingJobs = &(data_.pendingJobs);

      args->pendingJobsMutex = &(data_.pendingJobsMutex);
      args->kernelMutex      = &(data_.kernelMutex);

      args->kernelLaunch = &(data_.kernelLaunch[p]);
      args->kernelArgs   = &(data_.kernelArgs[p]);

      pthread_create(&data_.tid[p], NULL, pthreadLimbo, args);
    }
  }

  template <>
  void device_t<Pthreads>::getEnvironmentVariables(){
    char *c_compiler = getenv("OCCA_PTHREADS_COMPILER");
    if(c_compiler != NULL)
      compiler = std::string(c_compiler);

    char *c_compilerFlags = getenv("OCCA_PTHREADS_COMPILER_FLAGS");
    if(c_compilerFlags != NULL)
      compilerFlags = std::string(c_compilerFlags);
  }

  template <>
  void device_t<Pthreads>::setCompiler(const std::string &compiler_){
    compiler = compiler_;
  }

  template <>
  void device_t<Pthreads>::setCompilerFlags(const std::string &compilerFlags_){
    compilerFlags = compilerFlags_;
  }

  template <>
  void device_t<Pthreads>::flush(){}

  template <>
  void device_t<Pthreads>::finish(){
    OCCA_EXTRACT_DATA(Pthreads, Device);

    // Fence local data (incase of out-of-socket updates)
    while(data_.pendingJobs)
      __asm__ __volatile__ ("lfence");
  }

  template <>
  stream device_t<Pthreads>::genStream(){
    return NULL;
  }

  template <>
  void device_t<Pthreads>::freeStream(stream s){}

  template <>
  tag device_t<Pthreads>::tagStream(){
    tag ret;

    ret.tagTime = currentTime();

    return ret;
  }

  template <>
  double device_t<Pthreads>::timeBetween(const tag &startTag, const tag &endTag){
    return (endTag.tagTime - startTag.tagTime);
  }

  template <>
  kernel_v* device_t<Pthreads>::buildKernelFromSource(const std::string &filename,
                                                      const std::string &functionName,
                                                      const kernelInfo &info_){
    kernel_v *k = new kernel_t<Pthreads>;
    k->dev = dev;
    k->buildFromSource(filename, functionName, info_);
    return k;
  }

  template <>
  kernel_v* device_t<Pthreads>::buildKernelFromBinary(const std::string &filename,
                                                      const std::string &functionName){
    kernel_v *k = new kernel_t<Pthreads>;
    k->dev = dev;
    k->buildFromBinary(filename, functionName);
    return k;
  }

  template <>
  memory_v* device_t<Pthreads>::malloc(const size_t bytes,
                                       void *source){
    memory_v *mem = new memory_t<Pthreads>;

    mem->dev    = dev;
    mem->handle = ::malloc(bytes);
    mem->size   = bytes;

    if(source != NULL)
      ::memcpy(mem->handle, source, bytes);

    return mem;
  }

  template <>
  void device_t<Pthreads>::free(){
    finish();

    OCCA_EXTRACT_DATA(Pthreads, Device);

    pthread_mutex_destroy( &(data_.pendingJobsMutex) );
    pthread_mutex_destroy( &(data_.kernelMutex) );

    ::free(data);
  }

  template <>
  int device_t<Pthreads>::simdWidth(){
    simdWidth_ = OCCA_SIMD_WIDTH;
    return OCCA_SIMD_WIDTH;
  }
  //==================================
};
