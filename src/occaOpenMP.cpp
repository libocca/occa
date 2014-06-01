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
    info.addDefine("OCCA_USING_CPU", 1);
    info.addDefine("OCCA_USING_GPU", 0);

    info.addDefine("OCCA_USING_OPENMP", 1);
    info.addDefine("OCCA_USING_OPENCL", 0);
    info.addDefine("OCCA_USING_CUDA"  , 0);

    info.addOCCAKeywords(occaOpenMPDefines);

    std::stringstream salt;
    salt << "OpenMP"
         << info.salt()
         << functionName;

    std::string cachedBinary = binaryIsCached(filename, salt.str());

    struct stat buffer;
    bool fileExists = (stat(cachedBinary.c_str(), &buffer) == 0);

    if(fileExists){
      std::cout << "Found cached binary of [" << filename << "] in [" << cachedBinary << "]\n";
      return buildFromBinary(cachedBinary, functionName);
    }

    data = ::_mm_malloc(sizeof(OpenMPKernelData_t), OCCA_MEM_ALIGN);

    std::string iCachedBinary = createIntermediateSource(filename,
                                                         cachedBinary,
                                                         info);

    std::stringstream command;

    command << dev->ompCompiler
            << " -o " << cachedBinary
            << " -x c++ -w -fPIC -shared"
            << ' '    << dev->ompCompilerFlags
            << ' '    << info.flags
            << ' '    << iCachedBinary;

    const std::string &sCommand = command.str();

    std::cout << sCommand << '\n';

    system(sCommand.c_str());

    void *dlHandle = dlopen(cachedBinary.c_str(), RTLD_NOW);

    OCCA_CHECK(dlHandle != NULL);

    OCCA_EXTRACT_DATA(OpenMP, Kernel);

    data_.handle = dlsym(dlHandle, functionName.c_str());

    OCCA_CHECK(data_.handle != NULL);

    return this;
  }

  template <>
  kernel_t<OpenMP>* kernel_t<OpenMP>::buildFromBinary(const std::string &filename,
                                                      const std::string &functionName_){
    data = ::_mm_malloc(sizeof(OpenMPKernelData_t), OCCA_MEM_ALIGN);
    OCCA_EXTRACT_DATA(OpenMP, Kernel);

    functionName = functionName_;

    void *dlHandle = dlopen(filename.c_str(), RTLD_NOW);
    data_.handle = dlsym(dlHandle, functionName.c_str());

    OCCA_CHECK(data_.handle != NULL);

    return this;
  }

  template <>
  int kernel_t<OpenMP>::preferredDimSize(){
    preferredDimSize_ = OCCA_SIMD_WIDTH;
    return OCCA_SIMD_WIDTH;
  }

  OCCA_OPENMP_KERNEL_OPERATOR_DEFINITIONS;

  template <>
  double kernel_t<OpenMP>::timeTaken(){
    const double &start = *((double*) startTime);
    const double &end   = *((double*) endTime);

    return 1.0e3*(end - start);
  }

  template <>
  void kernel_t<OpenMP>::free(){
    OCCA_EXTRACT_DATA(OpenMP, Kernel);

    dlclose(data_.handle);
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
                                  const size_t bytes,
                                  const size_t offset){
    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    memcpy(((char*) handle) + offset, source, bytes_);
  }

  template <>
  void memory_t<OpenMP>::copyFrom(const memory_v *source,
                                  const size_t bytes,
                                  const size_t offset){
    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    memcpy(((char*) handle) + offset, source->handle, bytes_);
  }

  template <>
  void memory_t<OpenMP>::copyTo(void *dest,
                                const size_t bytes,
                                const size_t offset){
    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    memcpy(dest, ((char*) handle) + offset, bytes_);
  }

  template <>
  void memory_t<OpenMP>::copyTo(memory_v *dest,
                                const size_t bytes,
                                const size_t offset){
    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    memcpy(dest->handle, ((char*) handle) + offset, bytes_);
  }

  template <>
  void memory_t<OpenMP>::asyncCopyFrom(const void *source,
                                       const size_t bytes,
                                       const size_t offset){
    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    memcpy(((char*) handle) + offset, source , bytes_);
  }

  template <>
  void memory_t<OpenMP>::asyncCopyFrom(const memory_v *source,
                                       const size_t bytes,
                                       const size_t offset){
    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    memcpy(((char*) handle) + offset, source->handle , bytes_);
  }

  template <>
  void memory_t<OpenMP>::asyncCopyTo(void *dest,
                                     const size_t bytes,
                                     const size_t offset){
    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    memcpy(dest, ((char*) handle) + offset, bytes_);
  }

  template <>
  void memory_t<OpenMP>::asyncCopyTo(memory_v *dest,
                                     const size_t bytes,
                                     const size_t offset){
    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    memcpy(dest->handle, ((char*) handle) + offset, bytes_);
  }

  template <>
  void memory_t<OpenMP>::free(){
    ::free(handle);
  }
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<OpenMP>::device_t(){
    data = NULL;
    memoryUsed = 0;
  }

  template <>
  device_t<OpenMP>::device_t(int platform, int device){
    data       = NULL;
    memoryUsed = 0;
  }

  template <>
  device_t<OpenMP>::device_t(const device_t<OpenMP> &d){
    data       = d.data;
    memoryUsed = d.memoryUsed;
  }

  template <>
  device_t<OpenMP>& device_t<OpenMP>::operator = (const device_t<OpenMP> &d){
    data       = d.data;
    memoryUsed = d.memoryUsed;

    return *this;
  }

  template <>
  void device_t<OpenMP>::setup(const int platform, const int device){}

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
  memory_v* device_t<OpenMP>::malloc(const size_t bytes,
                                     void *source){
    memory_v *mem = new memory_t<OpenMP>;

    mem->dev    = dev;
    mem->handle = ::_mm_malloc(bytes, OCCA_MEM_ALIGN);
    mem->size   = bytes;

    if(source != NULL)
      ::memcpy(mem->handle, source, bytes);

    return mem;
  }

  template <>
  int device_t<OpenMP>::simdWidth(){
    simdWidth_ = OCCA_SIMD_WIDTH;
    return OCCA_SIMD_WIDTH;
  }
  //==================================
};
