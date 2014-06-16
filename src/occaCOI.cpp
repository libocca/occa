#if OCCA_COI_ENABLED

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

    preferredDimSize_ = 0;

    startTime = (void*) new coiEvent;
    endTime   = (void*) new coiEvent;
  }

  template <>
  kernel_t<COI>::kernel_t(const kernel_t<COI> &k){
    data = k.data;
    dev  = k.dev;

    functionName = k.functionName;

    dims  = k.dims;
    inner = k.inner;
    outer = k.outer;

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

    preferredDimSize_ = k.preferredDimSize_;

    *((coiEvent*) startTime) = *((coiEvent*) k.startTime);
    *((coiEvent*) endTime)   = *((coiEvent*) k.endTime);

    return *this;
  }

  template <>
  kernel_t<COI>::~kernel_t(){}

  template <>
  kernel_t<COI>* kernel_t<COI>::buildFromSource(const std::string &filename,
                                                const std::string &functionName_,
                                                const kernelInfo &info_){
    OCCA_EXTRACT_DATA(COI, Kernel);

    functionName = functionName_;

    kernelInfo info = info_;
    info.addDefine("OCCA_USING_CPU", 1);
    info.addDefine("OCCA_USING_COI", 1);

    info.addOCCAKeywords(occaCOIDefines);

    std::stringstream salt;
    salt << "COI"
         << data_.platform << '-' << data_.device
         << info.salt()
         << functionName;

    std::string cachedBinary = binaryIsCached(filename, salt.str());

    struct stat buffer;
    const bool fileExists = (stat(cachedBinary.c_str(), &buffer) == 0);

    if(fileExists){
      std::cout << "Found cached binary of [" << filename << "] in [" << cachedBinary << "]\n";
      return buildFromBinary(cachedBinary, functionName);
    }

    std::string iCachedBinary = createIntermediateSource(filename,
                                                         cachedBinary,
                                                         info);

    cl_int error;

    int fileHandle = ::open(iCachedBinary.c_str(), O_RDWR);
    if(fileHandle == 0)
      printf("File [ %s ] does not exist.\n", iCachedBinary.c_str());

    struct stat fileInfo;
    const int status = fstat(fileHandle, &fileInfo);

    if(status != 0)
      printf( "File [ %s ] gave a bad fstat.\n" , iCachedBinary.c_str());

    const size_t cLength = fileInfo.st_size;

    char *cFunction = new char[cLength + 1];

    ::read(fileHandle, cFunction, cLength);

    ::close(fileHandle);

    data_.program = clCreateProgramWithSource(data_.context, 1, (const char **) &cFunction, &cLength, &error);
    OCCA_CL_CHECK("Kernel (" + functionName + ") : Constructing Program", error);

    std::string catFlags = info.flags + dev->dHandle->compilerFlags;

    error = clBuildProgram(data_.program,
                           1, &data_.deviceID,
                           catFlags.c_str(),
                           NULL, NULL);

    if(error){
      cl_int error;
      char *log;
      size_t logSize;

      clGetProgramBuildInfo(data_.program, data_.deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

      if(logSize > 2){
        log = new char[logSize+1];

        error = clGetProgramBuildInfo(data_.program, data_.deviceID, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        OCCA_CL_CHECK("Kernel (" + functionName + ") : Building Program", error);
        log[logSize] = '\0';

        printf("Kernel (%s): Build Log\n%s", functionName.c_str(), log);

        delete[] log;
      }
    }

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Building Program", error);

    {
      size_t binarySize;
      char *binary;

      OCCA_CL_CHECK("saveProgramBinary: Getting Binary Sizes",
                    clGetProgramInfo(data_.program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binarySize, NULL));

      binary = new char[binarySize + 1];

      OCCA_CL_CHECK("saveProgramBinary: Getting Binary",
                    clGetProgramInfo(data_.program, CL_PROGRAM_BINARIES, sizeof(char*), &binary, NULL));

      FILE *fp = fopen(cachedBinary.c_str(), "wb");
      fwrite(binary, 1, binarySize, fp);
      fclose(fp);

      delete [] binary;
    }

    data_.kernel = clCreateKernel(data_.program, functionName.c_str(), &error);
    OCCA_CL_CHECK("Kernel (" + functionName + "): Creating Kernel", error);

    std::cout << "COI compiled " << filename << " from [" << iCachedBinary << "]\n";

    delete [] cFunction;

    return this;
  }

  template <>
  kernel_t<COI>* kernel_t<COI>::buildFromBinary(const std::string &filename,
                                                const std::string &functionName_){
    OCCA_EXTRACT_DATA(COI, Kernel);

    functionName = functionName_;

    cl_int binaryError, error;

    int fileHandle = ::open(filename.c_str(), O_RDWR);
    if(fileHandle == 0)
      printf("File [ %s ] does not exist.\n", filename.c_str());

    struct stat fileInfo;
    const int status = fstat(fileHandle, &fileInfo);

    if(status != 0)
      printf( "File [ %s ] gave a bad fstat.\n" , filename.c_str());

    const size_t fileSize = fileInfo.st_size;

    unsigned char *cFile = new unsigned char[fileSize];

    ::read(fileHandle, cFile, fileSize);

    ::close(fileHandle);

    data_.program = clCreateProgramWithBinary(data_.context,
                                              1, &(data_.deviceID),
                                              &fileSize,
                                              (const unsigned char**) &cFile,
                                              &binaryError, &error);
    OCCA_CL_CHECK("Kernel (" + functionName + ") : Constructing Program", binaryError);
    OCCA_CL_CHECK("Kernel (" + functionName + ") : Constructing Program", error);

    error = clBuildProgram(data_.program,
                           1, &data_.deviceID,
                           dev->dHandle->compilerFlags.c_str(),
                           NULL, NULL);

    if(error){
      cl_int error;
      char *log;
      size_t logSize;

      clGetProgramBuildInfo(data_.program, data_.deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

      if(logSize > 2){
        log = new char[logSize+1];

        error = clGetProgramBuildInfo(data_.program, data_.deviceID, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        OCCA_CL_CHECK("Kernel (" + functionName + ") : Building Program", error);
        log[logSize] = '\0';

        printf("Kernel (%s): Build Log\n%s", functionName.c_str(), log);

        delete[] log;
      }
    }

    OCCA_CL_CHECK("Kernel (" + functionName + ") : Building Program", error);

    data_.kernel = clCreateKernel(data_.program, functionName.c_str(), &error);
    OCCA_CL_CHECK("Kernel (" + functionName + "): Creating Kernel", error);

    delete [] cFile;

    return this;
  }

  template <>
  int kernel_t<COI>::preferredDimSize(){
    if(preferredDimSize_)
      return preferredDimSize_;

    OCCA_EXTRACT_DATA(COI, Kernel);

    size_t pds;

    OCCA_CL_CHECK("Kernel: Getting Preferred Dim Size",
                  clGetKernelWorkGroupInfo(data_.kernel,
                                           data_.deviceID,
                                           CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                           sizeof(size_t), &pds, NULL));

    preferredDimSize_ = pds;

    return preferredDimSize_;
  }

  OCCA_COI_KERNEL_OPERATOR_DEFINITIONS;

  template <>
  double kernel_t<COI>::timeTaken(){
    coiEvent &startEvent = *((coiEvent*) startTime);
    coiEvent &endEvent   = *((coiEvent*) endTime);

    cl_ulong start, end;

    clGetEventProfilingInfo(startEvent, CL_PROFILING_COMMAND_END,
                            sizeof(cl_ulong), &start,
                            NULL);

    clGetEventProfilingInfo(endEvent, CL_PROFILING_COMMAND_START,
                            sizeof(cl_ulong), &end,
                            NULL);

    return 1.0e-9*(end - start);
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
  }

  template <>
  memory_t<COI>::memory_t(const memory_t<COI> &m){
    handle = m.handle;
    dev    = m.dev;
    size   = m.size;
  }

  template <>
  memory_t<COI>& memory_t<COI>::operator = (const memory_t<COI> &m){
    handle = m.handle;
    dev    = m.dev;
    size   = m.size;

    return *this;
  }

  template <>
  memory_t<COI>::~memory_t(){}

  template <>
  void memory_t<COI>::copyFrom(const void *source,
                               const size_t bytes,
                               const size_t offset){
    coiStream &stream = *((coiStream*) dev->currentStream);

    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    coiEvent lastEvent = stream.lastEvent;

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferCopy(*((coiMemory*) handle),
                                 offset,
                                 source,
                                 bytes,
                                 COI_COPY_UNSPECIFIED,
                                 lastEvent != NULL,
                                 &lastEvent,
                                 &(stream.lastEvent)) );
  }

  template <>
  void memory_t<COI>::copyFrom(const memory_v *source,
                               const size_t bytes,
                               const size_t destOffset,
                               const size_t srcOffset){
    coiStream &stream = *((coiStream*) dev->currentStream);

    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    coiEvent lastEvent = stream.lastEvent;

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferCopy(*((coiMemory*) handle),
                                 *((coiMemory*) source->handle),
                                 destOffset,
                                 srcOffset,
                                 bytes,
                                 COI_COPY_UNSPECIFIED,
                                 lastEvent != NULL,
                                 &lastEvent,
                                 &(stream.lastEvent)) );
  }

  template <>
  void memory_t<COI>::copyTo(void *dest,
                             const size_t bytes,
                             const size_t offset){
    coiStream &stream = *((coiStream*) dev->currentStream);

    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    coiEvent lastEvent = stream.lastEvent;

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferCopy(*((coiMemory*) handle),
                                 offset,
                                 dest,
                                 bytes,
                                 COI_COPY_UNSPECIFIED,
                                 lastEvent != NULL,
                                 &lastEvent,
                                 &(stream.lastEvent)) );
  }

  template <>
  void memory_t<COI>::copyTo(memory_v *dest,
                             const size_t bytes,
                             const size_t destOffset,
                             const size_t srcOffset){
    coiStream &stream = *((coiStream*) dev->currentStream);

    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    coiEvent lastEvent = stream.lastEvent;

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferCopy(*((coiMemory*) dest->handle),
                                 *((coiMemory*) handle),
                                 destOffset,
                                 srcOffset,
                                 bytes,
                                 COI_COPY_UNSPECIFIED,
                                 lastEvent != NULL,
                                 &lastEvent,
                                 &(stream.lastEvent)) );
  }

  template <>
  void memory_t<COI>::asyncCopyFrom(const void *source,
                                    const size_t bytes,
                                    const size_t offset){
    coiStream &stream = *((coiStream*) dev->currentStream);

    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferCopy(*((coiMemory*) handle),
                                 offset,
                                 source,
                                 bytes,
                                 COI_COPY_UNSPECIFIED,
                                 0, NULL, NULL) );
  }

  template <>
  void memory_t<COI>::asyncCopyFrom(const memory_v *source,
                                    const size_t bytes,
                                    const size_t destOffset,
                                    const size_t srcOffset){
    coiStream &stream = *((coiStream*) dev->currentStream);

    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    coiEvent lastEvent = stream.lastEvent;

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferCopy(*((coiMemory*) handle),
                                 *((coiMemory*) source->handle),
                                 destOffset,
                                 srcOffset,
                                 bytes,
                                 COI_COPY_UNSPECIFIED,
                                 0, NULL, NULL) );
  }

  template <>
  void memory_t<COI>::asyncCopyTo(void *dest,
                                  const size_t bytes,
                                  const size_t offset){
    coiStream &stream = *((coiStream*) dev->currentStream);

    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    coiEvent lastEvent = stream.lastEvent;

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferCopy(*((coiMemory*) handle),
                                 offset,
                                 dest,
                                 bytes,
                                 COI_COPY_UNSPECIFIED,
                                 0, NULL, NULL) );
  }

  template <>
  void memory_t<COI>::asyncCopyTo(memory_v *dest,
                                  const size_t bytes,
                                  const size_t destOffset,
                                  const size_t srcOffset){
    coiStream &stream = *((coiStream*) dev->currentStream);

    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    coiEvent lastEvent = stream.lastEvent;

    OCCA_COI_CHECK("Memory: Copy From",
                   COIBufferCopy(*((coiMemory*) dest->handle),
                                 *((coiMemory*) handle),
                                 destOffset,
                                 srcOffset,
                                 bytes,
                                 COI_COPY_UNSPECIFIED,
                                 0, NULL, NULL) );
  }

  template <>
  void memory_t<COI>::free(){
    OCCA_COI_CHECK("Memory: free",
                   COIBufferDestroy( *((coiMemory*) handle) ) );

    ::free(handle);
  }
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<COI>::device_t() :
    memoryUsed(0) {
    data = NULL;

    getEnvironmentVariables();
  }

  template <>
  device_t<COI>::device_t(int platform, int device) :
    memoryUsed(0) {
    data = NULL;

    getEnvironmentVariables();
  }

  template <>
  device_t<COI>::device_t(const device_t<COI> &d){
    data       = d.data;
    memoryUsed = d.memoryUsed;

    compilerFlags = d.compilerFlags;
  }

  template <>
  device_t<COI>& device_t<COI>::operator = (const device_t<COI> &d){
    data       = d.data;
    memoryUsed = d.memoryUsed;

    compilerFlags = d.compilerFlags;

    return *this;
  }

  template <>
  void device_t<COI>::setup(const int device, const int memoryAllocated){
    data = ::malloc(sizeof(COIDeviceData_t));

    OCCA_EXTRACT_DATA(COI, Device);

    uint32_t deviceCount;
    OCCA_COI_CHECK("Device: Get Count",
                   COIEngineGetCount(COI_ISA_MIC, &deviceCount));

    OCCA_CHECK(device < deviceCount);

    OCCA_COI_CHECK("Device: Get Handle",
                   COIEngineGetHandle(COI_ISA_MIC, device, &data_.deviceID) );

    // [-] Need a simple basic main(argc, argv) binary

    OCCA_COI_CHECK("Device: Initializing",
                   COIProcessCreateFromFile(data_.deviceID,
                                            SINK_NAME,
                                            0    , NULL,
                                            false, NULL,
                                            true , NULL,
                                            memoryAllocated ? memoryAllocated : (4 << 30), // 4 GB
                                            NULL,
                                            &data_.chiefID) );
  }

  template <>
  void device_t<COI>::getEnvironmentVariables(){
    char *c_compilerFlags = getenv("OCCA_COI_COMPILER_FLAGS");
    if(c_compilerFlags != NULL)
      compilerFlags = std::string(c_compilerFlags);
  }

  template <>
  void device_t<COI>::setCompiler(const std::string &compiler_){
    compiler = compiler_;
  }

  template <>
  void device_t<COI>::setCompilerFlags(const std::string &compilerFlags_){
    compilerFlags = compilerFlags_;
  }

  template <>
  void device_t<COI>::flush(){}

  template <>
  void device_t<COI>::finish(){
    coiStream &stream = *((coiStream*) dev->currentStream);

    if(stream.lastEvent != NULL)
      OCCA_COI_CHECK("Device: Waiting for Event",
                     COIEventWait(1, stream.lastEvent,
                                  -1, true, NULL, NULL) );
  }

  template <>
  stream device_t<COI>::genStream(){
    OCCA_EXTRACT_DATA(COI, Device);

    coiStream *retStream = (coiStream*) ::malloc(sizeof(coiStream));

    OCCA_COI_CHECK("Device: Generating a Stream",
                   COIPipelineCreate(data_.chiefID,
                                     NULL, 0,
                                     &(retStream->handle)) );

    retStream->lastEvent = NULL;

    return retStream;
  }

  template <>
  void device_t<COI>::freeStream(stream s){
    coiStream *stream = (coiStream*) s;

    COIPipelineDestroy("Device: Freeing a Stream",
                       stream.handle);

    ::free(stream);
  }

  template <>
  kernel_v* device_t<COI>::buildKernelFromSource(const std::string &filename,
                                                 const std::string &functionName,
                                                 const kernelInfo &info_){
    OCCA_EXTRACT_DATA(COI, Device);

    kernel_v *k = new kernel_t<COI>;

    k->dev  = dev;
    k->data = ::malloc(sizeof(COIKernelData_t));

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
    k->data = ::malloc(sizeof(COIKernelData_t));

    COIKernelData_t &kData_ = *((COIKernelData_t*) k->data);

    kData_.chiefID = data_.chiefID;

    k->buildFromBinary(filename, functionName);
    return k;
  }

  template <>
  memory_v* device_t<COI>::malloc(const size_t bytes,
                                  void *source){
    OCCA_EXTRACT_DATA(COI, Device);

    memory_v *mem = new memory_t<COI>;
    cl_int error;

    mem->dev    = dev;
    mem->handle = ::malloc(sizeof(coiMemory));
    mem->size   = bytes;

    OCCA_COI_CHECK("Device: Malloc",
                   COIBufferCreate(bytes,
                                   COI_BUFFER_NORMAL,
                                   COI_SINK_MEMORY,
                                   source,
                                   1,
                                   &(data_.chiefID),
                                   (coiMemory*) mem->handle) );

    return mem;
  }

  template <>
  void device_t<COI>::free(){
    OCCA_EXTRACT_DATA(COI, Device);

    OCCA_COI_CHECK( COIProcessDestroy(data_.chiefID,
                                      -1,
                                      false,
                                      NULL,
                                      NULL ));

    ::free(data);
  }

  template <>
  int device_t<COI>::simdWidth(){
    simdWidth_ = 16; // [-] AVX-512

    return 16;
  }
  //==================================


  //---[ Error Handling ]-------------
  std::string coiError(int e){
    return std::string( COIResultGetName(e) );
  }
  //==================================
};

#endif
