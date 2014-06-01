#if OCCA_OPENCL_ENABLED

#include "occaOpenCL.hpp"

namespace occa {
  //---[ Kernel ]---------------------
  template <>
  kernel_t<OpenCL>::kernel_t(){
    data = NULL;
    dev  = NULL;

    functionName = "";

    dims  = 1;
    inner = occa::dim(1,1,1);
    outer = occa::dim(1,1,1);

    preferredDimSize_ = 0;

    startTime = (void*) new cl_event;
    endTime   = (void*) new cl_event;
  }

  template <>
  kernel_t<OpenCL>::kernel_t(const kernel_t<OpenCL> &k){
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
  kernel_t<OpenCL>& kernel_t<OpenCL>::operator = (const kernel_t<OpenCL> &k){
    data = k.data;
    dev  = k.dev;

    functionName = k.functionName;

    dims  = k.dims;
    inner = k.inner;
    outer = k.outer;

    preferredDimSize_ = k.preferredDimSize_;

    *((cl_event*) startTime) = *((cl_event*) k.startTime);
    *((cl_event*) endTime)   = *((cl_event*) k.endTime);

    return *this;
  }

  template <>
  kernel_t<OpenCL>::~kernel_t(){}

  template <>
  kernel_t<OpenCL>* kernel_t<OpenCL>::buildFromSource(const std::string &filename,
                                                      const std::string &functionName_,
                                                      const kernelInfo &info_){
    OCCA_EXTRACT_DATA(OpenCL, Kernel);

    functionName = functionName_;

    kernelInfo info = info_;
    info.addDefine("OCCA_USING_CPU", 0);
    info.addDefine("OCCA_USING_GPU", 1);

    info.addDefine("OCCA_USING_OPENMP", 0);
    info.addDefine("OCCA_USING_OPENCL", 1);
    info.addDefine("OCCA_USING_CUDA"  , 0);

    info.addOCCAKeywords(occaOpenCLDefines);

    std::stringstream salt;
    salt << "OpenCL"
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

    char *cFunction = (char*) malloc(cLength);

    ::read(fileHandle, cFunction, cLength);

    ::close(fileHandle);

    data_.program = clCreateProgramWithSource(data_.context, 1, (const char **) &cFunction, &cLength, &error);
    OCCA_CL_CHECK("Kernel (" + functionName + ") : Constructing Program", error);

    error = clBuildProgram(data_.program, 1, &data_.deviceID, info.flags.c_str(), NULL, NULL);

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
                   clGetProgramInfo(data_.program, CL_PROGRAM_BINARY_SIZES, sizeof(dim), &binarySize, NULL));

      binary = new char[binarySize];

      OCCA_CL_CHECK("saveProgramBinary: Getting Binary",
                   clGetProgramInfo(data_.program, CL_PROGRAM_BINARIES, sizeof(char*), &binary, NULL));

      FILE *fp = fopen(cachedBinary.c_str(), "wb");
      fwrite(binary, 1, binarySize, fp);
      fclose(fp);

      delete [] binary;
    }

    data_.kernel = clCreateKernel(data_.program, functionName.c_str(), &error);
    OCCA_CL_CHECK("Kernel (" + functionName + "): Creating Kernel", error);

    std::cout << "OpenCL compiled " << filename << " from [" << iCachedBinary << "]\n";

    delete [] cFunction;

    return this;
  }

  template <>
  kernel_t<OpenCL>* kernel_t<OpenCL>::buildFromBinary(const std::string &filename,
                                                      const std::string &functionName_){
    OCCA_EXTRACT_DATA(OpenCL, Kernel);

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

    unsigned char *cFile = (unsigned char*) malloc(fileSize);

    ::read(fileHandle, cFile, fileSize);

    ::close(fileHandle);

    data_.program = clCreateProgramWithBinary(data_.context,
                                              1, &(data_.deviceID),
                                              &fileSize,
                                              (const unsigned char**) &cFile,
                                              &binaryError, &error);
    OCCA_CL_CHECK("Kernel (" + functionName + ") : Constructing Program", binaryError);
    OCCA_CL_CHECK("Kernel (" + functionName + ") : Constructing Program", error);

    error = clBuildProgram(data_.program, 1, &data_.deviceID, NULL, NULL, NULL); // <> Needs flags!

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
  int kernel_t<OpenCL>::preferredDimSize(){
    if(preferredDimSize_)
      return preferredDimSize_;

    OCCA_EXTRACT_DATA(OpenCL, Kernel);

    size_t pds;

    OCCA_CL_CHECK("Kernel: Getting Preferred Dim Size",
                  clGetKernelWorkGroupInfo(data_.kernel,
                                           data_.deviceID,
                                           CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                           sizeof(size_t), &pds, NULL));

    preferredDimSize_ = pds;

    return preferredDimSize_;
  }

  OCCA_OPENCL_KERNEL_OPERATOR_DEFINITIONS;

  template <>
  double kernel_t<OpenCL>::timeTaken(){
    cl_event &startEvent = *((cl_event*) startTime);
    cl_event &endEvent   = *((cl_event*) endTime);

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
  void kernel_t<OpenCL>::free(){
  }
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<OpenCL>::memory_t(){
    handle = NULL;
    dev    = NULL;
    size = 0;
  }

  template <>
  memory_t<OpenCL>::memory_t(const memory_t<OpenCL> &m){
    handle = m.handle;
    dev    = m.dev;
    size   = m.size;
  }

  template <>
  memory_t<OpenCL>& memory_t<OpenCL>::operator = (const memory_t<OpenCL> &m){
    handle = m.handle;
    dev    = m.dev;
    size   = m.size;

    return *this;
  }

  template <>
  memory_t<OpenCL>::~memory_t(){}

  template <>
  void memory_t<OpenCL>::copyFrom(const void *source,
                                  const size_t bytes,
                                  const size_t offset){
    cl_command_queue &stream = *((cl_command_queue*) dev->currentStream);

    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    OCCA_CL_CHECK("Memory: Copy From",
                  clEnqueueWriteBuffer(stream, *((cl_mem*) handle),
                                       CL_TRUE,
                                       offset, bytes_, source,
                                       0, NULL, NULL));
  }

  template <>
  void memory_t<OpenCL>::copyFrom(const memory_v *source,
                                  const size_t bytes,
                                  const size_t offset){
    cl_command_queue &stream = *((cl_command_queue*) dev->currentStream);

    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    OCCA_CL_CHECK("Memory: Copy From",
                  clEnqueueCopyBuffer(stream,
                                      *((cl_mem*) source->handle),
                                      *((cl_mem*) handle),
                                      0, offset,// <>
                                      bytes_,
                                      0, NULL, NULL));
  }

  template <>
  void memory_t<OpenCL>::copyTo(void *dest,
                                const size_t bytes,
                                const size_t offset){
    const cl_command_queue &stream = *((cl_command_queue*) dev->currentStream);

    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    OCCA_CL_CHECK("Memory: Copy To",
                  clEnqueueReadBuffer(stream, *((cl_mem*) handle),
                                      CL_TRUE,
                                      offset, bytes_, dest,
                                      0, NULL, NULL));
  }

  template <>
  void memory_t<OpenCL>::copyTo(memory_v *dest,
                                const size_t bytes,
                                const size_t offset){
    const cl_command_queue &stream = *((cl_command_queue*) dev->currentStream);

    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    OCCA_CL_CHECK("Memory: Copy To",
                  clEnqueueCopyBuffer(stream,
                                      *((cl_mem*) handle),
                                      *((cl_mem*) dest->handle),
                                      offset, 0,// <>
                                      bytes_,
                                      0, NULL, NULL));
  }

  template <>
  void memory_t<OpenCL>::asyncCopyFrom(const void *source,
                                       const size_t bytes,
                                       const size_t offset){
    const cl_command_queue &stream = *((cl_command_queue*) dev->currentStream);

    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    OCCA_CL_CHECK("Memory: Asynchronous Copy From",
                  clEnqueueWriteBuffer(stream, *((cl_mem*) handle),
                                       CL_FALSE,
                                       offset, bytes_, source,
                                       0, NULL, NULL));
  }

  template <>
  void memory_t<OpenCL>::asyncCopyFrom(const memory_v *source,
                                       const size_t bytes,
                                       const size_t offset){
    const cl_command_queue &stream = *((cl_command_queue*) dev->currentStream);

    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    OCCA_CL_CHECK("Memory: Asynchronous Copy From",
                  clEnqueueCopyBuffer(stream,
                                      *((cl_mem*) source->handle),
                                      *((cl_mem*) handle),
                                      0, offset,// <>
                                      bytes_,
                                      0, NULL, NULL));
  }

  template <>
  void memory_t<OpenCL>::asyncCopyTo(void *dest,
                                     const size_t bytes,
                                     const size_t offset){
    const cl_command_queue &stream = *((cl_command_queue*) dev->currentStream);

    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    OCCA_CL_CHECK("Memory: Asynchronous Copy To",
                  clEnqueueReadBuffer(stream, *((cl_mem*) handle),
                                      CL_FALSE,
                                      offset, bytes_, dest,
                                      0, NULL, NULL));
  }

  template <>
  void memory_t<OpenCL>::asyncCopyTo(memory_v *dest,
                                     const size_t bytes,
                                     const size_t offset){
    const cl_command_queue &stream = *((cl_command_queue*) dev->currentStream);

    const size_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    OCCA_CL_CHECK("Memory: Asynchronous Copy To",
                  clEnqueueCopyBuffer(stream,
                                      *((cl_mem*) handle),
                                      *((cl_mem*) dest->handle),
                                      offset, 0, // <>
                                      bytes_,
                                      0, NULL, NULL));
  }

  template <>
  void memory_t<OpenCL>::free(){
    clReleaseMemObject(*((cl_mem*) handle));
    ::free(handle);
  }
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<OpenCL>::device_t() :
    data(NULL),
    memoryUsed(0) {}

  template <>
  device_t<OpenCL>::device_t(int platform, int device) :
    data(NULL),
    memoryUsed(0) {}

  template <>
  device_t<OpenCL>::device_t(const device_t<OpenCL> &d){
    data       = d.data;
    memoryUsed = d.memoryUsed;
  }

  template <>
  device_t<OpenCL>& device_t<OpenCL>::operator = (const device_t<OpenCL> &d){
    data       = d.data;
    memoryUsed = d.memoryUsed;

    return *this;
  }

  template <>
  void device_t<OpenCL>::setup(const int platform, const int device){
    data = ::_mm_malloc(sizeof(OpenCLDeviceData_t), OCCA_MEM_ALIGN);

    OCCA_EXTRACT_DATA(OpenCL, Device);
    cl_int error;

    data_.platform = platform;
    data_.device   = device;

    cl_platform_id *platforms = new cl_platform_id[platform + 1];
    cl_device_id   *devices   = new cl_device_id[device + 1];

    OCCA_CL_CHECK("OpenCL: Get Platform IDs",
                  clGetPlatformIDs(platform + 1, platforms, NULL));

    data_.platformID = platforms[platform];

    clGetDeviceIDs(data_.platformID,
                   CL_DEVICE_TYPE_ALL,
                   device + 1, devices, NULL);

    data_.deviceID = devices[device];

    data_.context = clCreateContext(NULL, 1, &data_.deviceID, NULL, NULL, &error);
    OCCA_CL_CHECK("Device: Creating Context", error);

    delete [] platforms;
    delete [] devices;
  }

  template <>
  void device_t<OpenCL>::flush(){
    clFlush(*((cl_command_queue*) dev->currentStream));
  }

  template <>
  void device_t<OpenCL>::finish(){
    clFinish(*((cl_command_queue*) dev->currentStream));
  }

  template <>
  stream device_t<OpenCL>::genStream(){
    OCCA_EXTRACT_DATA(OpenCL, Device);
    cl_int error;

    cl_command_queue *retStream = (cl_command_queue*) ::_mm_malloc(sizeof(cl_command_queue), OCCA_MEM_ALIGN);

    *retStream = clCreateCommandQueue(data_.context, data_.deviceID, CL_QUEUE_PROFILING_ENABLE, &error);
    OCCA_CL_CHECK("Device: genStream", error);

    return retStream;
  }

  template <>
  void device_t<OpenCL>::freeStream(stream s){
    OCCA_CL_CHECK("Device: freeStream",
                  clReleaseCommandQueue( *((cl_command_queue*) s) ));
    ::free(s);
  }

  template <>
  kernel_v* device_t<OpenCL>::buildKernelFromSource(const std::string &filename,
                                                   const std::string &functionName,
                                                   const kernelInfo &info_){
    OCCA_EXTRACT_DATA(OpenCL, Device);

    kernel_v *k = new kernel_t<OpenCL>;

    k->dev  = dev;
    k->data = ::_mm_malloc(sizeof(OpenCLKernelData_t), OCCA_MEM_ALIGN);

    OpenCLKernelData_t &kData_ = *((OpenCLKernelData_t*) k->data);

    kData_.platform = data_.platform;
    kData_.device   = data_.device;

    kData_.platformID = data_.platformID;
    kData_.deviceID   = data_.deviceID;
    kData_.context    = data_.context;

    k->buildFromSource(filename, functionName, info_);
    return k;
  }

  template <>
  kernel_v* device_t<OpenCL>::buildKernelFromBinary(const std::string &filename,
                                                    const std::string &functionName){
    OCCA_EXTRACT_DATA(OpenCL, Device);

    kernel_v *k = new kernel_t<OpenCL>;

    k->dev  = dev;
    k->data = ::_mm_malloc(sizeof(OpenCLKernelData_t), OCCA_MEM_ALIGN);

    OpenCLKernelData_t &kData_ = *((OpenCLKernelData_t*) k->data);

    kData_.platform = data_.platform;
    kData_.device   = data_.device;

    kData_.platformID = data_.platformID;
    kData_.deviceID   = data_.deviceID;
    kData_.context    = data_.context;

    k->buildFromBinary(filename, functionName);
    return k;
  }

  template <>
  memory_v* device_t<OpenCL>::malloc(const size_t bytes,
                                     void *source){
    OCCA_EXTRACT_DATA(OpenCL, Device);

    memory_v *mem = new memory_t<OpenCL>;
    cl_int error;

    mem->dev    = dev;
    mem->handle = ::_mm_malloc(sizeof(cl_mem), OCCA_MEM_ALIGN);
    mem->size   = bytes;

    if(source == NULL)
      *((cl_mem*) mem->handle) = clCreateBuffer(data_.context,
                                                CL_MEM_READ_WRITE,
                                                bytes, NULL, &error);
    else
      *((cl_mem*) mem->handle) = clCreateBuffer(data_.context,
                                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                bytes, source, &error);

    OCCA_CL_CHECK("Device: malloc", error);

    return mem;
  }

  template <>
  int device_t<OpenCL>::simdWidth(){
    if(simdWidth_)
      return simdWidth_;

    OCCA_EXTRACT_DATA(OpenCL, Device);

    cl_device_type dBuffer;
    bool isGPU;

    const int bSize = 8192;
    char buffer[bSize + 1];
    buffer[bSize] = '\0';

    OCCA_CL_CHECK("Device: DEVICE_TYPE",
                  clGetDeviceInfo(data_.deviceID, CL_DEVICE_TYPE, sizeof(dBuffer), &dBuffer, NULL));

    OCCA_CL_CHECK("Device: DEVICE_VENDOR",
                  clGetDeviceInfo(data_.deviceID, CL_DEVICE_VENDOR, bSize, buffer, NULL));

    if(dBuffer & CL_DEVICE_TYPE_CPU)
      isGPU = false;
    else if(dBuffer & CL_DEVICE_TYPE_GPU)
      isGPU = true;
    else{
      OCCA_CHECK(false);
    }

    if(isGPU){
      std::string vendor = buffer;
      if(vendor.find("NVIDIA") != std::string::npos)
        simdWidth_ = 32;
      else if((vendor.find("AMD")                    != std::string::npos) ||
              (vendor.find("Advanced Micro Devices") != std::string::npos))
        simdWidth_ = 64;
      else if(vendor.find("Intel") != std::string::npos)   // <> Need to check for Xeon Phi
        simdWidth_ = OCCA_SIMD_WIDTH;
      else{
        OCCA_CHECK(false);
      }
    }
    else
      simdWidth_ = OCCA_SIMD_WIDTH;

    return simdWidth_;
  }
  //==================================


  //---[ Error Handling ]-------------
  std::string openclError(int e){
    switch(e){
    case CL_SUCCESS:                                   return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:                          return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:                      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:                    return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:             return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:                          return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:                        return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:              return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:                          return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:                     return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:                return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:                     return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:                               return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:              return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case CL_INVALID_VALUE:                             return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:                       return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:                          return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:                            return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:                           return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:                  return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:                     return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:                          return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:                        return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:           return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:                        return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:                           return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:                            return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:                     return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:                           return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:                return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:                       return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:                 return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:                            return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:                         return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:                         return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:                          return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:                       return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:                    return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:                   return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:                    return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:                     return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:                   return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:                             return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:                         return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:                         return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:                       return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:                         return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:                  return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_PROPERTY:                          return "CL_INVALID_PROPERTY";
    default:                                           return "UNKNOWN ERROR";
    };
  }
  //==================================
};

#endif
