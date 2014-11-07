#if OCCA_OPENCL_ENABLED

#include "occaOpenCL.hpp"

namespace occa {
  //---[ Helper Functions ]-----------
  namespace cl {
    cl_device_type deviceType(int type){
      cl_device_type ret = 0;

      if(type & occa::CPU)     ret |= CL_DEVICE_TYPE_CPU;
      if(type & occa::GPU)     ret |= CL_DEVICE_TYPE_GPU;
      if(type & occa::FPGA)    ret |= CL_DEVICE_TYPE_ACCELERATOR;
      if(type & occa::XeonPhi) ret |= CL_DEVICE_TYPE_ACCELERATOR;

      return ret;
    }

    int platformCount(){
      cl_uint platformCount;

      OCCA_CL_CHECK("OpenCL: Get Platform ID Count",
                    clGetPlatformIDs(0, NULL, &platformCount));

      return platformCount;
    }

    cl_platform_id platformID(int pID){
      cl_platform_id *platforms = new cl_platform_id[pID + 1];

      OCCA_CL_CHECK("OpenCL: Get Platform ID",
                    clGetPlatformIDs(pID + 1, platforms, NULL));

      cl_platform_id ret = platforms[pID];

      delete [] platforms;

      return ret;
    }

    int deviceCount(int type){
      int pCount = cl::platformCount();
      int ret = 0;

      for(int p = 0; p < pCount; ++p)
        ret += deviceCountInPlatform(p, type);

      return ret;
    }

    int deviceCountInPlatform(int pID, int type){
      cl_uint dCount;

      cl_platform_id clPID = platformID(pID);

      OCCA_CL_CHECK("OpenCL: Get Device ID Count",
                    clGetDeviceIDs(clPID,
                                   deviceType(type),
                                   0, NULL, &dCount));

      return dCount;
    }

    cl_device_id deviceID(int pID, int dID, int type){
      cl_device_id *devices = new cl_device_id[dID + 1];

      cl_platform_id clPID = platformID(pID);

      OCCA_CL_CHECK("OpenCL: Get Device ID Count",
                    clGetDeviceIDs(clPID,
                                   deviceType(type),
                                   dID + 1, devices, NULL));

      cl_device_id ret = devices[dID];

      delete [] devices;

      return ret;
    }

    std::string deviceStrInfo(cl_device_id clDID,
                              cl_device_info clInfo){
      size_t bytes;

      OCCA_CL_CHECK("OpenCL: Getting Device String Info",
                    clGetDeviceInfo(clDID,
                                    clInfo,
                                    0, NULL, &bytes));

      char *buffer  = new char[bytes + 1];
      buffer[bytes] = '\0';

      OCCA_CL_CHECK("OpenCL: Getting Device String Info",
                    clGetDeviceInfo(clDID,
                                    clInfo,
                                    bytes, buffer, NULL));

      std::string ret = buffer;

      delete [] buffer;

      int firstNS, lastNS;

      for(int i = 0; i < ret.size(); ++i){
        if((ret[i] != ' ') &&
           (ret[i] != '\t') &&
           (ret[i] != '\n')){
          firstNS = i;
          break;
        }
      }

      for(int i = (ret.size() - 1); i > firstNS; --i){
        if((ret[i] != ' ') &&
           (ret[i] != '\t') &&
           (ret[i] != '\n')){
          lastNS = i;
          break;
        }
      }

      return ret.substr(firstNS, (lastNS - firstNS + 1));
    }

    std::string deviceName(int pID, int dID){
      cl_device_id clDID = deviceID(pID, dID);

      return deviceStrInfo(clDID, CL_DEVICE_NAME);
    }

    int deviceType(int pID, int dID){
      cl_device_id clDID = deviceID(pID, dID);
      int ret = 0;

      cl_device_type clDeviceType;

      OCCA_CL_CHECK("OpenCL: Get Device Type",
                    clGetDeviceInfo(clDID,
                                    CL_DEVICE_TYPE,
                                    sizeof(clDeviceType), &clDeviceType, NULL));

      if(clDeviceType & CL_DEVICE_TYPE_CPU)
        ret |= occa::CPU;
      else if(clDeviceType & CL_DEVICE_TYPE_GPU)
        ret |= occa::GPU;

      return ret;
    }

    int deviceVendor(int pID, int dID){
      cl_device_id clDID = deviceID(pID, dID);
      int ret = 0;

      std::string vendor = deviceStrInfo(clDID, CL_DEVICE_VENDOR);

      if(vendor.find("AMD")                    != std::string::npos ||
         vendor.find("Advanced Micro Devices") != std::string::npos ||
         vendor.find("ATI")                    != std::string::npos)
        ret |= occa::AMD;

      else if(vendor.find("Intel") != std::string::npos)
        ret |= occa::Intel;

      else if(vendor.find("Altera") != std::string::npos)
        ret |= occa::Altera;

      else if(vendor.find("Nvidia") != std::string::npos ||
              vendor.find("NVIDIA") != std::string::npos)
        ret |= occa::NVIDIA;

      return ret;
    }

    int deviceCoreCount(int pID, int dID){
      cl_device_id clDID = deviceID(pID, dID);
      cl_uint ret;

      OCCA_CL_CHECK("OpenCL: Get Device Core Count",
                    clGetDeviceInfo(clDID,
                                    CL_DEVICE_MAX_COMPUTE_UNITS,
                                    sizeof(ret), &ret, NULL));

      return ret;
    }

    occa::deviceInfo deviceInfo(int pID, int dID){
      occa::deviceInfo dInfo;

      dInfo.name  = deviceName(pID, dID);
      dInfo.id    = pID*100 + dID;
      dInfo.info  = deviceType(pID, dID) | deviceVendor(pID, dID) | occa::OpenCL;

      if((dInfo.info & occa::GPU) && (dInfo.info & occa::AMD))
        dInfo.preferredMode = occa::OpenCL;

      return dInfo;
    }

    void buildKernelFromSource(OpenCLKernelData_t &data_,
                               const char *content,
                               const size_t contentBytes,
                               const std::string &functionName,
                               const std::string &flags,
                               const std::string &cachedBinary,
                               const std::string &iCachedBinary){
      cl_int error;

      data_.program = clCreateProgramWithSource(data_.context, 1, (const char **) &content, &contentBytes, &error);

      if(error && cachedBinary.size())
        releaseFile(cachedBinary);

      if(iCachedBinary.size()){
        std::cout << "OpenCL compiling " << functionName
                  << " from [" << iCachedBinary << "]";

        if(flags.size())
          std::cout << " with flags [" << flags << "]";

        std::cout << '\n';
      }
      else
        std::cout << "OpenCL compiling " << functionName << " from [Library]\n";

      OCCA_CL_CHECK("Kernel (" + functionName + ") : Constructing Program", error);

      error = clBuildProgram(data_.program,
                             1, &data_.deviceID,
                             flags.c_str(),
                             NULL, NULL);

      if(error){
        cl_int error;
        char *log;
        uintptr_t logSize;

        clGetProgramBuildInfo(data_.program, data_.deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

        if(logSize > 2){
          log = new char[logSize+1];

          error = clGetProgramBuildInfo(data_.program, data_.deviceID, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
          OCCA_CL_CHECK("Kernel (" + functionName + ") : Building Program", error);
          log[logSize] = '\0';

          printf("Kernel (%s): Build Log\n%s", functionName.c_str(), log);

          delete[] log;
        }

        if(cachedBinary.size())
          releaseFile(cachedBinary);
      }

      OCCA_CL_CHECK("Kernel (" + functionName + ") : Building Program", error);

      data_.kernel = clCreateKernel(data_.program, functionName.c_str(), &error);

      if(error && cachedBinary.size())
        releaseFile(cachedBinary);

      OCCA_CL_CHECK("Kernel (" + functionName + "): Creating Kernel", error);

      if(iCachedBinary.size()){
        std::cout << "OpenCL compiled " << functionName << " from [" << iCachedBinary << "]";

        if(flags.size())
          std::cout << " with flags [" << flags << "]";

        std::cout << '\n';
      }
      else
        std::cout << "OpenCL compiled " << functionName << " from [Library]\n";
    }

    void buildKernelFromBinary(OpenCLKernelData_t &data_,
                               const unsigned char *content,
                               const size_t contentBytes,
                               const std::string &functionName,
                               const std::string &flags){
      cl_int error, binaryError;

      data_.program = clCreateProgramWithBinary(data_.context,
                                                1, &(data_.deviceID),
                                                &contentBytes,
                                                (const unsigned char**) &content,
                                                &binaryError, &error);

      OCCA_CL_CHECK("Kernel (" + functionName + ") : Constructing Program", binaryError);
      OCCA_CL_CHECK("Kernel (" + functionName + ") : Constructing Program", error);

      error = clBuildProgram(data_.program,
                             1, &data_.deviceID,
                             flags.c_str(),
                             NULL, NULL);

      if(error){
        cl_int error;
        char *log;
        uintptr_t logSize;

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
    }

    void saveProgramBinary(OpenCLKernelData_t &data_,
                           const std::string &cachedBinary){
      uintptr_t binarySize;
      char *binary;

      cl_int error = clGetProgramInfo(data_.program, CL_PROGRAM_BINARY_SIZES, sizeof(uintptr_t), &binarySize, NULL);

      if(error)
        releaseFile(cachedBinary);

      OCCA_CL_CHECK("saveProgramBinary: Getting Binary Sizes", error);

      binary = new char[binarySize + 1];

      error = clGetProgramInfo(data_.program, CL_PROGRAM_BINARIES, sizeof(char*), &binary, NULL);

      if(error)
        releaseFile(cachedBinary);

      OCCA_CL_CHECK("saveProgramBinary: Getting Binary", error);

      FILE *fp = fopen(cachedBinary.c_str(), "wb");
      fwrite(binary, 1, binarySize, fp);
      fclose(fp);

      delete [] binary;
    }

    occa::device wrapDevice(cl_platform_id platformID,
                            cl_device_id deviceID,
                            cl_context context){
      occa::device dev;
      device_t<OpenCL> &devH      = *(new device_t<OpenCL>());
      OpenCLDeviceData_t &devData = *(new OpenCLDeviceData_t);

      dev.mode_   = occa::OpenCL;
      dev.strMode = "OpenCL";
      dev.dHandle = &devH;

      // This will get set in the copy-back
      devH.dev = &dev;

      //---[ Setup ]----------
      devH.data = &devData;

      devData.platform = -1;
      devData.device   = -1;

      devData.platformID = platformID;
      devData.deviceID   = deviceID;
      devData.context    = context;
      //======================

      dev.modelID_ = library::deviceModelID(dev.getIdentifier());
      dev.id_      = library::genDeviceID();

      dev.currentStream = dev.genStream();

      return dev;
    }
  };

  const cl_channel_type clFormats[8] = {CL_UNSIGNED_INT8,
                                        CL_UNSIGNED_INT16,
                                        CL_UNSIGNED_INT32,
                                        CL_SIGNED_INT8,
                                        CL_SIGNED_INT16,
                                        CL_SIGNED_INT32,
                                        CL_HALF_FLOAT,
                                        CL_FLOAT};

  template <>
  void* formatType::format<occa::OpenCL>() const {
    return ((void*) &(clFormats[format_]));
  }
  //==================================


  //---[ Kernel ]---------------------
  template <>
  kernel_t<OpenCL>::kernel_t(){
    data = NULL;
    dev  = NULL;

    functionName = "";

    dims  = 1;
    inner = occa::dim(1,1,1);
    outer = occa::dim(1,1,1);

    nestedKernelCount = 0;

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

    nestedKernelCount = k.nestedKernelCount;

    if(nestedKernelCount){
      nestedKernels = new kernel[nestedKernelCount];

      for(int i = 0; i < nestedKernelCount; ++i)
        nestedKernels[i] = k.nestedKernels[i];
    }

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

    nestedKernelCount = k.nestedKernelCount;

    if(nestedKernelCount){
      nestedKernels = new kernel[nestedKernelCount];

      for(int i = 0; i < nestedKernelCount; ++i)
        nestedKernels[i] = k.nestedKernels[i];
    }

    preferredDimSize_ = k.preferredDimSize_;

    *((cl_event*) startTime) = *((cl_event*) k.startTime);
    *((cl_event*) endTime)   = *((cl_event*) k.endTime);

    return *this;
  }

  template <>
  kernel_t<OpenCL>::~kernel_t(){}

  template <>
  std::string kernel_t<OpenCL>::getCachedBinaryName(const std::string &filename,
                                                    kernelInfo &info_){
    OCCA_EXTRACT_DATA(OpenCL, Kernel);

    info_.addDefine("OCCA_USING_GPU"   , 1);
    info_.addDefine("OCCA_USING_OPENCL", 1);

    info_.addOCCAKeywords(occaOpenCLDefines);

    std::stringstream salt;

    salt << "OpenCL"
         << data_.platform << '-' << data_.device
         << info_.salt()
         << parser::version
         << dev->dHandle->compilerEnvScript
         << dev->dHandle->compiler
         << dev->dHandle->compilerFlags;

    return getCachedName(filename, salt.str());
  }

  template <>
  kernel_t<OpenCL>* kernel_t<OpenCL>::buildFromSource(const std::string &filename,
                                                      const std::string &functionName_,
                                                      const kernelInfo &info_){
    OCCA_EXTRACT_DATA(OpenCL, Kernel);

    functionName = functionName_;

    kernelInfo info = info_;
    std::string cachedBinary = getCachedBinaryName(filename, info);

    struct stat buffer;
    const bool fileExists = (stat(cachedBinary.c_str(), &buffer) == 0);

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

    cl_int error;

    int fileHandle = ::open(iCachedBinary.c_str(), O_RDWR);
    if(fileHandle == 0){
      printf("File [ %s ] does not exist.\n", iCachedBinary.c_str());
      releaseFile(cachedBinary);
      throw 1;
    }

    struct stat fileInfo;
    const int status = fstat(fileHandle, &fileInfo);

    if(status != 0){
      printf( "File [ %s ] gave a bad fstat.\n" , iCachedBinary.c_str());
      releaseFile(cachedBinary);
      throw 1;
    }

    const uintptr_t cLength = fileInfo.st_size;

    char *cFunction = new char[cLength + 1];
    cFunction[cLength] = '\0';

    ::read(fileHandle, cFunction, cLength);

    ::close(fileHandle);

    std::string catFlags = info.flags + dev->dHandle->compilerFlags;

    cl::buildKernelFromSource(data_,
                              cFunction, cLength,
                              functionName,
                              catFlags,
                              cachedBinary, iCachedBinary);

    cl::saveProgramBinary(data_, cachedBinary);

    releaseFile(cachedBinary);

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
    if(fileHandle == 0){
      printf("File [ %s ] does not exist.\n", filename.c_str());
      throw 1;
    }

    struct stat fileInfo;
    const int status = fstat(fileHandle, &fileInfo);

    if(status != 0){
      printf( "File [ %s ] gave a bad fstat.\n" , filename.c_str());
      throw 1;
    }

    const uintptr_t fileSize = fileInfo.st_size;

    unsigned char *cFile = new unsigned char[fileSize + 1];
    cFile[fileSize] = '\0';

    ::read(fileHandle, cFile, fileSize);

    ::close(fileHandle);

    cl::buildKernelFromBinary(data_,
                              cFile, fileSize,
                              functionName,
                              dev->dHandle->compilerFlags);

    delete [] cFile;

    return this;
  }

  template <>
  kernel_t<OpenCL>* kernel_t<OpenCL>::loadFromLibrary(const char *cache,
                                                      const std::string &functionName_){
    OCCA_EXTRACT_DATA(OpenCL, Kernel);

    functionName = functionName_;

    cl::buildKernelFromSource(data_,
                              cache, strlen(cache),
                              functionName_);

    return this;
  }

  template <>
  int kernel_t<OpenCL>::preferredDimSize(){
    if(preferredDimSize_)
      return preferredDimSize_;

    OCCA_EXTRACT_DATA(OpenCL, Kernel);

    uintptr_t pds;

    OCCA_CL_CHECK("Kernel: Getting Preferred Dim Size",
                  clGetKernelWorkGroupInfo(data_.kernel,
                                           data_.deviceID,
                                           CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                           sizeof(uintptr_t), &pds, NULL));

    preferredDimSize_ = pds;

    return preferredDimSize_;
  }

#include "operators/occaOpenCLKernelOperators.cpp"

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
  double kernel_t<OpenCL>::timeTakenBetween(void *start, void *end){
    cl_event &startEvent = *((cl_event*) start);
    cl_event &endEvent   = *((cl_event*) end);

    cl_ulong start_, end_;

    clGetEventProfilingInfo(startEvent, CL_PROFILING_COMMAND_END,
                            sizeof(cl_ulong), &start_,
                            NULL);

    clGetEventProfilingInfo(endEvent, CL_PROFILING_COMMAND_START,
                            sizeof(cl_ulong), &end_,
                            NULL);

    return 1.0e-9*(end_ - start_);
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

    isTexture = false;
    textureInfo.dim = 1;
    textureInfo.w = textureInfo.h = textureInfo.d = 0;

    isAWrapper = false;
  }

  template <>
  memory_t<OpenCL>::memory_t(const memory_t<OpenCL> &m){
    handle = m.handle;
    dev    = m.dev;
    size   = m.size;

    isTexture = m.isTexture;
    textureInfo.dim  = m.textureInfo.dim;

    textureInfo.w = m.textureInfo.w;
    textureInfo.h = m.textureInfo.h;
    textureInfo.d = m.textureInfo.d;

    isAWrapper = m.isAWrapper;
  }

  template <>
  memory_t<OpenCL>& memory_t<OpenCL>::operator = (const memory_t<OpenCL> &m){
    handle = m.handle;
    dev    = m.dev;
    size   = m.size;

    isTexture = m.isTexture;
    textureInfo.dim  = m.textureInfo.dim;

    textureInfo.w = m.textureInfo.w;
    textureInfo.h = m.textureInfo.h;
    textureInfo.d = m.textureInfo.d;

    isAWrapper = m.isAWrapper;

    return *this;
  }

  template <>
  memory_t<OpenCL>::~memory_t(){}

  template <>
  void* memory_t<OpenCL>::getMemoryHandle(){
    return handle;
  }

  template <>
  void* memory_t<OpenCL>::getTextureHandle(){
    return textureInfo.arg;
  }

  template <>
  void memory_t<OpenCL>::copyFrom(const void *source,
                                  const uintptr_t bytes,
                                  const uintptr_t offset){
    cl_command_queue &stream = *((cl_command_queue*) dev->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    if(!isTexture)
      OCCA_CL_CHECK("Memory: Copy From",
                    clEnqueueWriteBuffer(stream, *((cl_mem*) handle),
                                         CL_TRUE,
                                         offset, bytes_, source,
                                         0, NULL, NULL));
    else{
      const int bie = textureInfo.bytesInEntry;

      size_t offset_[3] = {      (offset / bie)      % textureInfo.w , 0, 0};
      size_t pixels_[3] = {1 + (((bytes_ / bie) - 1) % textureInfo.w), 1, 1};

      if(textureInfo.dim == 2){
        offset_[1] = (offset / bie) / textureInfo.w;
        pixels_[1] = (bytes_ / bie) / textureInfo.w;
      }

      OCCA_CL_CHECK("Texture Memory: Copy From",
                    clEnqueueWriteImage(stream, *((cl_mem*) handle),
                                        CL_TRUE,
                                        offset_, pixels_,
                                        0, 0,
                                        source,
                                        0, NULL, NULL));
    }
  }

  template <>
  void memory_t<OpenCL>::copyFrom(const memory_v *source,
                                  const uintptr_t bytes,
                                  const uintptr_t destOffset,
                                  const uintptr_t srcOffset){
    cl_command_queue &stream = *((cl_command_queue*) dev->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + destOffset) <= size);
    OCCA_CHECK((bytes_ + srcOffset)  <= source->size);

    if(!isTexture)
      OCCA_CL_CHECK("Memory: Copy From",
                    clEnqueueCopyBuffer(stream,
                                        *((cl_mem*) source->handle),
                                        *((cl_mem*) handle),
                                        srcOffset, destOffset,
                                        bytes_,
                                        0, NULL, NULL));
    else
      OCCA_CL_CHECK("Texture Memory: Copy From",
                    clEnqueueCopyBuffer(stream,
                                        *((cl_mem*) source->handle),
                                        *((cl_mem*) handle),
                                        srcOffset, destOffset,
                                        bytes_,
                                        0, NULL, NULL));
  }

  template <>
  void memory_t<OpenCL>::copyTo(void *dest,
                                const uintptr_t bytes,
                                const uintptr_t offset){
    const cl_command_queue &stream = *((cl_command_queue*) dev->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    if(!isTexture)
      OCCA_CL_CHECK("Memory: Copy To",
                    clEnqueueReadBuffer(stream, *((cl_mem*) handle),
                                        CL_TRUE,
                                        offset, bytes_, dest,
                                        0, NULL, NULL));
    else{
      const int bie = textureInfo.bytesInEntry;

      size_t offset_[3] = {      (offset / bie)      % textureInfo.w , 0, 0};
      size_t pixels_[3] = {1 + (((bytes_ / bie) - 1) % textureInfo.w), 1, 1};

      if(textureInfo.dim == 2){
        offset_[1] = (offset / bie) / textureInfo.w;
        pixels_[1] = (bytes_ / bie) / textureInfo.w;
      }

      for(int i = 0; i < 3; ++i)
        std::cout << "offset[" << i << "] = " << offset_[i] << '\n';

      for(int i = 0; i < 3; ++i)
        std::cout << "pixels[" << i << "] = " << pixels_[i] << '\n';

      OCCA_CL_CHECK("Texture Memory: Copy From",
                    clEnqueueReadImage(stream, *((cl_mem*) handle),
                                       CL_TRUE,
                                       offset_, pixels_,
                                       0, 0,
                                       dest,
                                       0, NULL, NULL));
    }
  }

  template <>
  void memory_t<OpenCL>::copyTo(memory_v *dest,
                                const uintptr_t bytes,
                                const uintptr_t destOffset,
                                const uintptr_t srcOffset){
    const cl_command_queue &stream = *((cl_command_queue*) dev->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + srcOffset)  <= size);
    OCCA_CHECK((bytes_ + destOffset) <= dest->size);

    if(!isTexture)
      OCCA_CL_CHECK("Memory: Copy To",
                    clEnqueueCopyBuffer(stream,
                                        *((cl_mem*) handle),
                                        *((cl_mem*) dest->handle),
                                        srcOffset, destOffset,
                                        bytes_,
                                        0, NULL, NULL));
    else
      OCCA_CL_CHECK("Texture Memory: Copy To",
                    clEnqueueCopyBuffer(stream,
                                        *((cl_mem*) handle),
                                        *((cl_mem*) dest->handle),
                                        srcOffset, destOffset,
                                        bytes_,
                                        0, NULL, NULL));
  }

  template <>
  void memory_t<OpenCL>::asyncCopyFrom(const void *source,
                                       const uintptr_t bytes,
                                       const uintptr_t offset){
    const cl_command_queue &stream = *((cl_command_queue*) dev->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    if(!isTexture)
      OCCA_CL_CHECK("Memory: Asynchronous Copy From",
                    clEnqueueWriteBuffer(stream, *((cl_mem*) handle),
                                         CL_FALSE,
                                         offset, bytes_, source,
                                         0, NULL, NULL));
    else
      OCCA_CL_CHECK("Texture Memory: Asynchronous Copy From",
                    clEnqueueWriteBuffer(stream, *((cl_mem*) handle),
                                         CL_FALSE,
                                         offset, bytes_, source,
                                         0, NULL, NULL));
  }

  template <>
  void memory_t<OpenCL>::asyncCopyFrom(const memory_v *source,
                                       const uintptr_t bytes,
                                       const uintptr_t destOffset,
                                       const uintptr_t srcOffset){
    const cl_command_queue &stream = *((cl_command_queue*) dev->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + destOffset) <= size);
    OCCA_CHECK((bytes_ + srcOffset)  <= source->size);

    if(!isTexture)
      OCCA_CL_CHECK("Memory: Asynchronous Copy From",
                    clEnqueueCopyBuffer(stream,
                                        *((cl_mem*) source->handle),
                                        *((cl_mem*) handle),
                                        srcOffset, destOffset,
                                        bytes_,
                                        0, NULL, NULL));
    else
      OCCA_CL_CHECK("Texture Memory: Asynchronous Copy From",
                    clEnqueueCopyBuffer(stream,
                                        *((cl_mem*) source->handle),
                                        *((cl_mem*) handle),
                                        srcOffset, destOffset,
                                        bytes_,
                                        0, NULL, NULL));
  }

  template <>
  void memory_t<OpenCL>::asyncCopyTo(void *dest,
                                     const uintptr_t bytes,
                                     const uintptr_t offset){
    const cl_command_queue &stream = *((cl_command_queue*) dev->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size);

    if(!isTexture)
      OCCA_CL_CHECK("Memory: Asynchronous Copy To",
                    clEnqueueReadBuffer(stream, *((cl_mem*) handle),
                                        CL_FALSE,
                                        offset, bytes_, dest,
                                        0, NULL, NULL));
    else
      OCCA_CL_CHECK("Texture Memory: Asynchronous Copy To",
                    clEnqueueReadBuffer(stream, *((cl_mem*) handle),
                                        CL_FALSE,
                                        offset, bytes_, dest,
                                        0, NULL, NULL));
  }

  template <>
  void memory_t<OpenCL>::asyncCopyTo(memory_v *dest,
                                     const uintptr_t bytes,
                                     const uintptr_t destOffset,
                                     const uintptr_t srcOffset){
    const cl_command_queue &stream = *((cl_command_queue*) dev->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + srcOffset)  <= size);
    OCCA_CHECK((bytes_ + destOffset) <= dest->size);

    if(!isTexture)
      OCCA_CL_CHECK("Memory: Asynchronous Copy To",
                    clEnqueueCopyBuffer(stream,
                                        *((cl_mem*) handle),
                                        *((cl_mem*) dest->handle),
                                        srcOffset, destOffset,
                                        bytes_,
                                        0, NULL, NULL));
    else
      OCCA_CL_CHECK("Texture Memory: Asynchronous Copy To",
                    clEnqueueCopyBuffer(stream,
                                        *((cl_mem*) handle),
                                        *((cl_mem*) dest->handle),
                                        srcOffset, destOffset,
                                        bytes_,
                                        0, NULL, NULL));
  }

  template <>
  void memory_t<OpenCL>::free(){
    clReleaseMemObject(*((cl_mem*) handle));

    if(!isAWrapper)
      delete (cl_mem*) handle;

    if(isTexture){
      clReleaseSampler( *((cl_sampler*) textureInfo.arg) );

      if(!isAWrapper)
        delete (cl_sampler*) textureInfo.arg;
    }

    size = 0;
  }
  //==================================


  //---[ Device ]---------------------
  template <>
  std::vector<occa::deviceInfo> availableDevices<OpenCL>(){
    std::vector<occa::deviceInfo> ret( occa::cl::deviceCount() );
    int pos = 0;

    int platformCount = occa::cl::platformCount();

    std::cout << occa::deviceInfo::sLine  << '\n'
              << occa::deviceInfo::header << '\n'
              << occa::deviceInfo::sLine  << '\n';

    for(int p = 0; p < platformCount; ++p){
      int deviceCount = occa::cl::deviceCountInPlatform(p);

      for(int d = 0; d < deviceCount; ++d){
        ret[pos] = occa::cl::deviceInfo(p,d);

        std::cout << ret[pos].summarizedInfo() << '\n';

        ++pos;
      }
    }

    std::cout << occa::deviceInfo::sLine << '\n';

    return ret;
  }

  template <>
  device_t<OpenCL>::device_t() {
    data            = NULL;
    memoryAllocated = 0;

    getEnvironmentVariables();
  }

  template <>
  device_t<OpenCL>::device_t(int platform, int device){
    data            = NULL;
    memoryAllocated = 0;

    getEnvironmentVariables();
  }

  template <>
  device_t<OpenCL>::device_t(const device_t<OpenCL> &d){
    data            = d.data;
    memoryAllocated = d.memoryAllocated;

    compilerFlags = d.compilerFlags;
  }

  template <>
  device_t<OpenCL>& device_t<OpenCL>::operator = (const device_t<OpenCL> &d){
    data            = d.data;
    memoryAllocated = d.memoryAllocated;

    compilerFlags = d.compilerFlags;

    return *this;
  }

  template <>
  void device_t<OpenCL>::setup(const int platform, const int device){
    data = new OpenCLDeviceData_t;

    OCCA_EXTRACT_DATA(OpenCL, Device);
    cl_int error;

    data_.platform = platform;
    data_.device   = device;

    data_.platformID = cl::platformID(platform);
    data_.deviceID   = cl::deviceID(platform, device);

    data_.context = clCreateContext(NULL, 1, &data_.deviceID, NULL, NULL, &error);
    OCCA_CL_CHECK("Device: Creating Context", error);
  }

  template <>
  deviceIdentifier device_t<OpenCL>::getIdentifier() const {
    deviceIdentifier dID;

    dID.mode_ = OpenCL;

    return dID;
  }

  template <>
  void device_t<OpenCL>::getEnvironmentVariables(){
    char *c_compilerFlags = getenv("OCCA_OPENCL_COMPILER_FLAGS");
    if(c_compilerFlags != NULL)
      compilerFlags = std::string(c_compilerFlags);
    else{
#if OCCA_DEBUG_ENABLED
      compilerFlags = "-cl-opt-disable";
#else
      compilerFlags = "";
#endif
    }
  }

  template <>
  void device_t<OpenCL>::appendAvailableDevices(std::vector<device> &dList){
    int platformCount = occa::cl::platformCount();

    for(int p = 0; p < platformCount; ++p){
      int deviceCount = occa::cl::deviceCountInPlatform(p);

      for(int d = 0; d < deviceCount; ++d){
        device dev;
        dev.setup("OpenCL", p, d);

        dList.push_back(dev);
      }
    }
  }

  template <>
  void device_t<OpenCL>::setCompiler(const std::string &compiler_){
    compiler = compiler_;
  }

  template <>
  void device_t<OpenCL>::setCompilerEnvScript(const std::string &compilerEnvScript_){
    compilerEnvScript = compilerEnvScript_;
  }

  template <>
  void device_t<OpenCL>::setCompilerFlags(const std::string &compilerFlags_){
    compilerFlags = compilerFlags_;
  }

  template <>
  std::string& device_t<OpenCL>::getCompiler(){
    return compiler;
  }

  template <>
  std::string& device_t<OpenCL>::getCompilerEnvScript(){
    return compilerEnvScript;
  }

  template <>
  std::string& device_t<OpenCL>::getCompilerFlags(){
    return compilerFlags;
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

    cl_command_queue *retStream = new cl_command_queue;

    *retStream = clCreateCommandQueue(data_.context, data_.deviceID, CL_QUEUE_PROFILING_ENABLE, &error);
    OCCA_CL_CHECK("Device: genStream", error);

    return retStream;
  }

  template <>
  void device_t<OpenCL>::freeStream(stream s){
    OCCA_CL_CHECK("Device: freeStream",
                  clReleaseCommandQueue( *((cl_command_queue*) s) ));
    delete (cl_command_queue*) s;
  }

  template <>
  stream device_t<OpenCL>::wrapStream(void *handle_){
    return handle_;
  }

  template <>
  tag device_t<OpenCL>::tagStream(){
    cl_command_queue &stream = *((cl_command_queue*) dev->currentStream);

    tag ret;

#ifdef CL_VERSION_1_2
    clEnqueueMarkerWithWaitList(stream, 0, NULL, &(ret.clEvent));
#else
    clEnqueueMarker(stream, &(ret.clEvent));
#endif

    return ret;
  }

  template <>
  double device_t<OpenCL>::timeBetween(const tag &startTag, const tag &endTag){
    cl_command_queue &stream = *((cl_command_queue*) dev->currentStream);
    cl_ulong start, end;

    clFinish(stream);

    OCCA_CL_CHECK ("Device: Time Between Tags (Start)",
                   clGetEventProfilingInfo(startTag.clEvent  ,
                                           CL_PROFILING_COMMAND_END,
                                           sizeof(cl_ulong),
                                           &start, NULL) );

    OCCA_CL_CHECK ("Device: Time Between Tags (End)",
                   clGetEventProfilingInfo(endTag.clEvent  ,
                                           CL_PROFILING_COMMAND_START,
                                           sizeof(cl_ulong),
                                           &end, NULL) );

    clReleaseEvent(startTag.clEvent);
    clReleaseEvent(endTag.clEvent);

    return (double) (1.0e-9 * (double)(end - start));
  }

  template <>
  kernel_v* device_t<OpenCL>::buildKernelFromSource(const std::string &filename,
                                                   const std::string &functionName,
                                                   const kernelInfo &info_){
    OCCA_EXTRACT_DATA(OpenCL, Device);

    kernel_v *k = new kernel_t<OpenCL>;

    k->dev  = dev;
    k->data = new OpenCLKernelData_t;

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
    k->data = new OpenCLKernelData_t;

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
  void device_t<OpenCL>::cacheKernelInLibrary(const std::string &filename,
                                              const std::string &functionName,
                                              const kernelInfo &info_){
    //---[ Creating shared library ]----
    OCCA_EXTRACT_DATA(OpenCL, Device);

    kernel tmpK = dev->buildKernelFromSource(filename, functionName, info_);
    tmpK.free();

    kernelInfo info = info_;
    info.addDefine("OCCA_USING_GPU"   , 1);
    info.addDefine("OCCA_USING_OPENCL", 1);

    info.addOCCAKeywords(occaOpenCLDefines);

    std::stringstream salt;
    salt << "OpenCL"
         << data_.platform << '-' << data_.device
         << info.salt()
         << parser::version
         << compilerEnvScript
         << compiler
         << compilerFlags;

    std::string cachedBinary = getCachedName(filename, salt.str());

    std::string prefix, name;
    getFilePrefixAndName(cachedBinary, prefix, name);

    std::string extension = getFileExtension(filename);

    const std::string iCachedBinary = prefix + "i_" + name;

    std::string contents = readFile(iCachedBinary);
    //==================================

    library::infoID_t infoID;

    infoID.modelID    = dev->modelID_;
    infoID.kernelName = functionName;

    library::infoHeader_t &header = library::headerMap[infoID];

    header.fileID = -1;
    header.mode   = OpenCL;

    const std::string flatDevID = getIdentifier().flattenFlagMap();

    header.flagsOffset = library::addToScratchPad(flatDevID);
    header.flagsBytes  = flatDevID.size();

    header.contentOffset = library::addToScratchPad(contents);
    header.contentBytes  = contents.size();

    header.kernelNameOffset = library::addToScratchPad(functionName);
    header.kernelNameBytes  = functionName.size();
  }

  template <>
  kernel_v* device_t<OpenCL>::loadKernelFromLibrary(const char *cache,
                                                    const std::string &functionName_){
    OCCA_EXTRACT_DATA(OpenCL, Device);

    kernel_v *k = new kernel_t<OpenCL>;

    k->dev  = dev;
    k->data = new OpenCLKernelData_t;

    OpenCLKernelData_t &kData_ = *((OpenCLKernelData_t*) k->data);

    kData_.platform = data_.platform;
    kData_.device   = data_.device;

    kData_.platformID = data_.platformID;
    kData_.deviceID   = data_.deviceID;
    kData_.context    = data_.context;

    k->loadFromLibrary(cache, functionName_);
    return k;
  }

  template <>
  memory_v* device_t<OpenCL>::wrapMemory(void *handle_,
                                         const uintptr_t bytes){
    memory_v *mem = new memory_t<OpenCL>;

    mem->dev    = dev;
    mem->size   = bytes;
    mem->handle = handle_;

    mem->isAWrapper = true;

    return mem;
  }

  template <>
  memory_v* device_t<OpenCL>::wrapTexture(void *handle_,
                                          const int dim, const occa::dim &dims,
                                          occa::formatType type, const int permissions){
#ifndef CL_VERSION_1_2
    if(dim == 1)
      return wrapMemory(handle_, dims.x * type.bytes());

    OCCA_EXTRACT_DATA(OpenCL, Device);

    memory_v *mem = new memory_t<OpenCL>;
    cl_int error;

    mem->dev    = dev;
    mem->size   = (dims.x * dims.y) * type.bytes();
    mem->handle = handle_;

    mem->isAWrapper = true;

    mem->isTexture = true;
    mem->textureInfo.dim  = dim;

    mem->textureInfo.w = dims.x;
    mem->textureInfo.h = dims.y;
    mem->textureInfo.d = dims.z;

    mem->textureInfo.bytesInEntry = type.bytes();

    mem->textureInfo.arg = new cl_sampler;

    *((cl_sampler*) mem->textureInfo.arg) = clCreateSampler(data_.context,
                                                            CL_FALSE,                 // Are args Normalized?
                                                            CL_ADDRESS_CLAMP_TO_EDGE, // Clamp edges
                                                            CL_FILTER_NEAREST,        // Point interpolation
                                                            &error);

    OCCA_CL_CHECK("Device: Creating texture sampler", error);

    return mem;
#else
    OCCA_EXTRACT_DATA(OpenCL, Device);
    cl_int error;

    memory_v *mem = new memory_t<OpenCL>;

    mem->dev    = dev;
    mem->size   = ((dim == 1) ? dims.x : (dims.x * dims.y)) * type.bytes();
    mem->handle = handle_;

    mem->isAWrapper = true;

    mem->isTexture = true;
    mem->textureInfo.dim  = dim;

    mem->textureInfo.w = dims.x;
    mem->textureInfo.h = dims.y;
    mem->textureInfo.d = dims.z;

    mem->textureInfo.bytesInEntry = type.bytes();

    mem->textureInfo.arg = new cl_sampler;

    *((cl_sampler*) mem->textureInfo.arg) = clCreateSampler(data_.context,
                                                            CL_FALSE,                 // Are args Normalized?
                                                            CL_ADDRESS_CLAMP_TO_EDGE, // Clamp edges
                                                            CL_FILTER_NEAREST,        // Point interpolation
                                                            &error);

    OCCA_CL_CHECK("Device: Creating texture sampler", error);

    return mem;
#endif
  }

  template <>
  memory_v* device_t<OpenCL>::malloc(const uintptr_t bytes,
                                     void *source){
    OCCA_EXTRACT_DATA(OpenCL, Device);

    memory_v *mem = new memory_t<OpenCL>;
    cl_int error;

    mem->dev    = dev;
    mem->handle = new cl_mem;
    mem->size   = bytes;

    if(source == NULL){
      *((cl_mem*) mem->handle) = clCreateBuffer(data_.context,
                                                CL_MEM_READ_WRITE,
                                                bytes, NULL, &error);
    }
    else{
      *((cl_mem*) mem->handle) = clCreateBuffer(data_.context,
                                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                bytes, source, &error);

      finish();
    }

    return mem;
  }

  template <>
  memory_v* device_t<OpenCL>::talloc(const int dim, const occa::dim &dims,
                                     void *source,
                                     occa::formatType type, const int permissions){
#ifndef CL_VERSION_1_2
    if(dim == 1)
      return malloc(dims.x * type.bytes(), source);

    OCCA_EXTRACT_DATA(OpenCL, Device);

    memory_v *mem = new memory_t<OpenCL>;
    cl_int error;

    mem->dev    = dev;
    mem->handle = new cl_mem;
    mem->size   = (dims.x * dims.y) * type.bytes();

    mem->isTexture = true;
    mem->textureInfo.dim  = dim;

    mem->textureInfo.w = dims.x;
    mem->textureInfo.h = dims.y;
    mem->textureInfo.d = dims.z;

    mem->textureInfo.bytesInEntry = type.bytes();

    cl_mem_flags flag = (CL_MEM_COPY_HOST_PTR |
                         ((permissions == occa::readWrite) ?
                          CL_MEM_READ_WRITE : CL_MEM_READ_ONLY));
    cl_image_format imageFormat;

    const int count = type.count();

    switch(count){
    case 1: imageFormat.image_channel_order = CL_R;    break;
    case 2: imageFormat.image_channel_order = CL_RG;   break;
    case 4: imageFormat.image_channel_order = CL_RGBA; break;
    };

    imageFormat.image_channel_data_type = *((cl_channel_type*) type.format<OpenCL>());

    *((cl_mem*) mem->handle) = clCreateImage2D(data_.context, flag,
                                               &imageFormat,
                                               dims.x,
                                               dims.y,
                                               0,
                                               source, &error);

    OCCA_CL_CHECK("Device: Allocating texture", error);

    mem->textureInfo.arg = new cl_sampler;

    *((cl_sampler*) mem->textureInfo.arg) = clCreateSampler(data_.context,
                                                            CL_FALSE,                 // Are args Normalized?
                                                            CL_ADDRESS_CLAMP_TO_EDGE, // Clamp edges
                                                            CL_FILTER_NEAREST,        // Point interpolation
                                                            &error);

    OCCA_CL_CHECK("Device: Creating texture sampler", error);

    return mem;
#else
    OCCA_EXTRACT_DATA(OpenCL, Device);

    memory_v *mem = new memory_t<OpenCL>;
    cl_int error;

    mem->dev    = dev;
    mem->handle = new cl_mem;
    mem->size   = ((dim == 1) ? dims.x : (dims.x * dims.y)) * type.bytes();

    mem->isTexture = true;
    mem->textureInfo.dim  = dim;

    mem->textureInfo.w = dims.x;
    mem->textureInfo.h = dims.y;
    mem->textureInfo.d = dims.z;

    mem->textureInfo.bytesInEntry = type.bytes();

    cl_mem_flags flag = (CL_MEM_COPY_HOST_PTR |
                         ((permissions == occa::readWrite) ?
                          CL_MEM_READ_WRITE : CL_MEM_READ_ONLY));
    cl_image_format imageFormat;
    cl_image_desc imageDesc;

    const int count = type.count();

    switch(count){
    case 1: imageFormat.image_channel_order = CL_R;    break;
    case 2: imageFormat.image_channel_order = CL_RG;   break;
    case 4: imageFormat.image_channel_order = CL_RGBA; break;
    };

    imageFormat.image_channel_data_type = *((cl_channel_type*) type.format<OpenCL>());

    imageDesc.image_type        = (dim == 1) ? CL_MEM_OBJECT_IMAGE1D : CL_MEM_OBJECT_IMAGE2D;
    imageDesc.image_width       = dims.x;
    imageDesc.image_height      = dims.y;
    imageDesc.image_array_size  = 1;
    imageDesc.image_row_pitch   = 0;
    imageDesc.image_slice_pitch = 0;
    imageDesc.num_mip_levels    = 0;
    imageDesc.num_samples       = 0;
    imageDesc.buffer            = NULL;

    *((cl_mem*) mem->handle) = clCreateImage(data_.context, flag,
                                             &imageFormat, &imageDesc,
                                             source, &error);

    OCCA_CL_CHECK("Device: Allocating texture", error);

    mem->textureInfo.arg = new cl_sampler;

    *((cl_sampler*) mem->textureInfo.arg) = clCreateSampler(data_.context,
                                                            CL_FALSE,                 // Are args Normalized?
                                                            CL_ADDRESS_CLAMP_TO_EDGE, // Clamp edges
                                                            CL_FILTER_NEAREST,        // Point interpolation
                                                            &error);

    OCCA_CL_CHECK("Device: Creating texture sampler", error);

    return mem;
#endif

    finish();
  }

  template <>
  void device_t<OpenCL>::free(){
    OCCA_EXTRACT_DATA(OpenCL, Device);

    OCCA_CL_CHECK("Device: Freeing Context",
                  clReleaseContext(data_.context) );

    delete (OpenCLDeviceData_t*) data;
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
