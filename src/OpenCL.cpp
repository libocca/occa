#if OCCA_OPENCL_ENABLED

#include "occa/OpenCL.hpp"

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

    int getPlatformCount(){
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

    int getDeviceCount(int type){
      int pCount = cl::getPlatformCount();
      int ret = 0;

      for(int p = 0; p < pCount; ++p)
        ret += getDeviceCountInPlatform(p, type);

      return ret;
    }

    int getDeviceCountInPlatform(int pID, int type){
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

      size_t firstNS = ret.size();
      size_t lastNS  = ret.size();

      size_t i;

      for(i = 0; i < ret.size(); ++i){
        if((ret[i] != ' ') &&
           (ret[i] != '\t') &&
           (ret[i] != '\n')){
          firstNS = i;
          break;
        }
      }

      if(i == ret.size())
        return "";

      for(i = (ret.size() - 1); i > firstNS; --i){
        if((ret[i] != ' ') &&
           (ret[i] != '\t') &&
           (ret[i] != '\n')){
          lastNS = i;
          break;
        }
      }

      if(i == firstNS)
        return "";

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
         vendor.find("ATI")                    != std::string::npos){

        ret |= occa::AMD;
      }
      else if(vendor.find("Intel") != std::string::npos){
        ret |= occa::Intel;
      }
      else if(vendor.find("Altera") != std::string::npos){
        ret |= occa::Altera;
      }
      else if(vendor.find("Nvidia") != std::string::npos ||
              vendor.find("NVIDIA") != std::string::npos){

        ret |= occa::NVIDIA;
      }

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

    uintptr_t getDeviceMemorySize(cl_device_id dID){
      cl_ulong ret;

      OCCA_CL_CHECK("OpenCL: Get Device Available Memory",
                    clGetDeviceInfo(dID,
                                    CL_DEVICE_GLOBAL_MEM_SIZE,
                                    sizeof(ret), &ret, NULL));

      return ret;
    }

    uintptr_t getDeviceMemorySize(int pID, int dID){
      cl_device_id clDID = deviceID(pID, dID);

      return getDeviceMemorySize(clDID);
    }

    std::string getDeviceListInfo(){
      std::stringstream ss;

      int platformCount = occa::cl::getPlatformCount();

      for(int pID = 0; pID < platformCount; ++pID){
        int deviceCount = occa::cl::getDeviceCountInPlatform(pID);

        for(int dID = 0; dID < deviceCount; ++dID){
          uintptr_t bytes      = getDeviceMemorySize(pID, dID);
          std::string bytesStr = stringifyBytes(bytes);

          if(pID || dID){
            ss << "              |-----------------------+------------------------------------------\n"
               << "              |  Device Name          | " << deviceName(pID, dID)            << '\n';
          }
          else{
            ss << "    OpenCL    |  Device Name          | " << deviceName(pID, dID)            << '\n';
          }

          ss << "              |  Driver Vendor        | " << occa::vendor(deviceVendor(pID,dID)) << '\n'
             << "              |  Platform ID          | " << pID                                 << '\n'
             << "              |  Device ID            | " << dID                                 << '\n'
             << "              |  Memory               | " << bytesStr                            << '\n';
        }
      }


      return ss.str();
    }

    void buildKernelFromSource(OpenCLKernelData_t &data_,
                               const char *content,
                               const size_t contentBytes,
                               const std::string &functionName,
                               const std::string &flags,
                               const std::string &hash,
                               const std::string &sourceFile){
      cl_int error;

      data_.program = clCreateProgramWithSource(data_.context, 1,
                                                (const char **) &content,
                                                &contentBytes,
                                                &error);

      if(error && hash.size())
        releaseHash(hash, 0);

      if(verboseCompilation_f){
        if(hash.size()){
          std::cout << "OpenCL compiling " << functionName
                    << " from [" << sourceFile << "]";

          if(flags.size())
            std::cout << " with flags [" << flags << "]";

          std::cout << '\n';
        }
        else
          std::cout << "OpenCL compiling " << functionName << " from [Library]\n";
      }

      OCCA_CL_CHECK("Kernel (" + functionName + ") : Constructing Program", error);

      error = clBuildProgram(data_.program,
                             1, &data_.deviceID,
                             flags.c_str(),
                             NULL, NULL);

      if(error){
        cl_int logError;
        char *log;
        uintptr_t logSize;

        clGetProgramBuildInfo(data_.program, data_.deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

        if(logSize > 2){
          log = new char[logSize+1];

          logError = clGetProgramBuildInfo(data_.program, data_.deviceID, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
          OCCA_CL_CHECK("Kernel (" + functionName + ") : Building Program", logError);
          log[logSize] = '\0';

          printf("Kernel (%s): Build Log\n%s", functionName.c_str(), log);

          delete[] log;
        }

        if(hash.size())
          releaseHash(hash, 0);
      }

      OCCA_CL_CHECK("Kernel (" + functionName + ") : Building Program", error);

      data_.kernel = clCreateKernel(data_.program, functionName.c_str(), &error);

      if(error && hash.size())
        releaseHash(hash, 0);

      OCCA_CL_CHECK("Kernel (" + functionName + "): Creating Kernel", error);

      if(verboseCompilation_f){
        if(sourceFile.size()){
          std::cout << "OpenCL compiled " << functionName << " from [" << sourceFile << "]";

          if(flags.size())
            std::cout << " with flags [" << flags << "]";

          std::cout << '\n';
        }
        else
          std::cout << "OpenCL compiled " << functionName << " from [Library]\n";
      }
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
        cl_int logError;
        char *log;
        uintptr_t logSize;

        clGetProgramBuildInfo(data_.program, data_.deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

        if(logSize > 2){
          log = new char[logSize+1];

          logError = clGetProgramBuildInfo(data_.program, data_.deviceID, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
          OCCA_CL_CHECK("Kernel (" + functionName + ") : Building Program", logError);
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
                           const std::string &binaryFile,
                           const std::string &hash){
      size_t binarySize;
      char *binary;

      cl_int error = clGetProgramInfo(data_.program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binarySize, NULL);

      if(error && hash.size())
        releaseHash(hash, 0);

      OCCA_CL_CHECK("saveProgramBinary: Getting Binary Sizes", error);

      binary = new char[binarySize + 1];

      error = clGetProgramInfo(data_.program, CL_PROGRAM_BINARIES, sizeof(char*), &binary, NULL);

      if(error && hash.size())
        releaseHash(hash, 0);

      OCCA_CL_CHECK("saveProgramBinary: Getting Binary", error);

      FILE *fp = fopen(binaryFile.c_str(), "wb");
      fwrite(binary, 1, binarySize, fp);
      fclose(fp);

      delete [] binary;
    }

    occa::device wrapDevice(cl_platform_id platformID,
                            cl_device_id deviceID,
                            cl_context context){
      occa::device dev;
      device_t<OpenCL> &dHandle   = *(new device_t<OpenCL>());
      OpenCLDeviceData_t &devData = *(new OpenCLDeviceData_t);

      dev.dHandle = &dHandle;

      //---[ Setup ]----------
      dHandle.data = &devData;

      devData.platform = -1;
      devData.device   = -1;

      devData.platformID = platformID;
      devData.deviceID   = deviceID;
      devData.context    = context;
      //======================

      dHandle.modelID_ = library::deviceModelID(dHandle.getIdentifier());
      dHandle.id_      = library::genDeviceID();

      dHandle.currentStream = dHandle.createStream();

      return dev;
    }

    bool imageFormatIsSupported(cl_image_format &f,
                                cl_image_format *fs,
                                const int formatCount){

      for(int i = 0; i < formatCount; ++i){
        cl_image_format &f2 = fs[i];

        const bool orderSupported = (f.image_channel_order ==
                                     (f.image_channel_order &
                                      f2.image_channel_order));

        const bool typeSupported = (f.image_channel_data_type ==
                                    (f.image_channel_data_type &
                                     f2.image_channel_data_type));

        if(orderSupported && typeSupported)
          return true;
      }

      return false;
    }

    void printImageFormat(cl_image_format &imageFormat){
      std::cout << "---[ OpenCL Image Format ]--------------\n"
                << "Supported Channel Formats:\n";

#define OCCA_CL_PRINT_CHANNEL_INFO(X)                                   \
      if(imageFormat.image_channel_order & X) std::cout << "  " #X "\n"

      OCCA_CL_PRINT_CHANNEL_INFO(CL_R);
      OCCA_CL_PRINT_CHANNEL_INFO(CL_Rx);
      OCCA_CL_PRINT_CHANNEL_INFO(CL_A);
      OCCA_CL_PRINT_CHANNEL_INFO(CL_INTENSITY);
      OCCA_CL_PRINT_CHANNEL_INFO(CL_RG);
      OCCA_CL_PRINT_CHANNEL_INFO(CL_RGx);
      OCCA_CL_PRINT_CHANNEL_INFO(CL_RA);
      OCCA_CL_PRINT_CHANNEL_INFO(CL_RGB);
      OCCA_CL_PRINT_CHANNEL_INFO(CL_RGBx);
      OCCA_CL_PRINT_CHANNEL_INFO(CL_RGBA);
      OCCA_CL_PRINT_CHANNEL_INFO(CL_ARGB);
      OCCA_CL_PRINT_CHANNEL_INFO(CL_BGRA);

#undef OCCA_CL_PRINT_CHANNEL_INFO

      std::cout << "\nSupported Channel Types:\n";

#define OCCA_CL_PRINT_CHANNEL_TYPE(X)                                   \
      if(imageFormat.image_channel_data_type & X) std::cout << "  " #X "\n"

      OCCA_CL_PRINT_CHANNEL_TYPE(CL_SNORM_INT8);
      OCCA_CL_PRINT_CHANNEL_TYPE(CL_SNORM_INT16);
      OCCA_CL_PRINT_CHANNEL_TYPE(CL_UNORM_INT8);
      OCCA_CL_PRINT_CHANNEL_TYPE(CL_UNORM_INT16);
      OCCA_CL_PRINT_CHANNEL_TYPE(CL_UNORM_SHORT_565);
      OCCA_CL_PRINT_CHANNEL_TYPE(CL_UNORM_SHORT_555);
      OCCA_CL_PRINT_CHANNEL_TYPE(CL_UNORM_INT_101010);
      OCCA_CL_PRINT_CHANNEL_TYPE(CL_SIGNED_INT8);
      OCCA_CL_PRINT_CHANNEL_TYPE(CL_SIGNED_INT16);
      OCCA_CL_PRINT_CHANNEL_TYPE(CL_SIGNED_INT32);
      OCCA_CL_PRINT_CHANNEL_TYPE(CL_UNSIGNED_INT8);
      OCCA_CL_PRINT_CHANNEL_TYPE(CL_UNSIGNED_INT16);
      OCCA_CL_PRINT_CHANNEL_TYPE(CL_UNSIGNED_INT32);
      OCCA_CL_PRINT_CHANNEL_TYPE(CL_HALF_FLOAT);
      OCCA_CL_PRINT_CHANNEL_TYPE(CL_FLOAT);

#undef OCCA_CL_PRINT_CHANNEL_TYPE

      std::cout << "========================================\n";
    }
  }

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
    strMode = "OpenCL";

    data    = NULL;
    dHandle = NULL;

    dims  = 1;
    inner = occa::dim(1,1,1);
    outer = occa::dim(1,1,1);

    maximumInnerDimSize_ = 0;
    preferredDimSize_    = 0;
  }

  template <>
  kernel_t<OpenCL>::kernel_t(const kernel_t<OpenCL> &k){
    *this = k;
  }

  template <>
  kernel_t<OpenCL>& kernel_t<OpenCL>::operator = (const kernel_t<OpenCL> &k){
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
  kernel_t<OpenCL>::~kernel_t(){}

  template <>
  void* kernel_t<OpenCL>::getKernelHandle(){
    OCCA_EXTRACT_DATA(OpenCL, Kernel);

    return data_.kernel;
  }

  template <>
  void* kernel_t<OpenCL>::getProgramHandle(){
    OCCA_EXTRACT_DATA(OpenCL, Kernel);

    return data_.program;
  }

  template <>
  std::string kernel_t<OpenCL>::fixBinaryName(const std::string &filename){
    return filename;
  }

  template <>
  kernel_t<OpenCL>* kernel_t<OpenCL>::buildFromSource(const std::string &filename,
                                                      const std::string &functionName,
                                                      const kernelInfo &info_){

    name = functionName;

    OCCA_EXTRACT_DATA(OpenCL, Kernel);

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

    createSourceFileFrom(filename, hashDir, info);

    std::string cFunction = readFile(sourceFile);

    std::string catFlags = info.flags + dHandle->compilerFlags;

    cl::buildKernelFromSource(data_,
                              cFunction.c_str(), cFunction.size(),
                              functionName,
                              catFlags,
                              hash, sourceFile);

    cl::saveProgramBinary(data_, binaryFile, hash);

    releaseHash(hash, 0);

    return this;
  }

  template <>
  kernel_t<OpenCL>* kernel_t<OpenCL>::buildFromBinary(const std::string &filename,
                                                      const std::string &functionName){

    name = functionName;

    OCCA_EXTRACT_DATA(OpenCL, Kernel);

    std::string cFile = readFile(filename);

    cl::buildKernelFromBinary(data_,
                              (const unsigned char*) cFile.c_str(),
                              cFile.size(),
                              functionName,
                              dHandle->compilerFlags);

    return this;
  }

  template <>
  kernel_t<OpenCL>* kernel_t<OpenCL>::loadFromLibrary(const char *cache,
                                                      const std::string &functionName){
    OCCA_EXTRACT_DATA(OpenCL, Kernel);

    cl::buildKernelFromSource(data_,
                              cache, strlen(cache),
                              functionName);

    return this;
  }

  template <>
  uintptr_t kernel_t<OpenCL>::maximumInnerDimSize(){
    if(maximumInnerDimSize_)
      return maximumInnerDimSize_;

    OCCA_EXTRACT_DATA(OpenCL, Kernel);

    size_t pds;

    OCCA_CL_CHECK("Kernel: Getting Preferred Dim Size",
                  clGetKernelWorkGroupInfo(data_.kernel,
                                           data_.deviceID,
                                           CL_KERNEL_WORK_GROUP_SIZE,
                                           sizeof(size_t), &pds, NULL));

    maximumInnerDimSize_ = (uintptr_t) pds;

    return maximumInnerDimSize_;
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

    preferredDimSize_ = (uintptr_t) pds;

    return preferredDimSize_;
  }

  template <>
  void kernel_t<OpenCL>::runFromArguments(const int kArgc, const kernelArg *kArgs){
    OpenCLKernelData_t &data_ = *((OpenCLKernelData_t*) data);
    cl_kernel kernel_ = data_.kernel;

    occa::dim fullOuter = outer*inner;

    int argc = 0;
    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [0]",
                  clSetKernelArg(kernel_, argc++, sizeof(void*), NULL));

    for(int i = 0; i < kArgc; ++i){
      for(int j = 0; j < kArgs[i].argc; ++j){
        OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Setting Kernel Argument [" << (i + 1) << "]",
                      clSetKernelArg(kernel_, argc++, kArgs[i].args[j].size, kArgs[i].args[j].ptr()));
      }
    }

    OCCA_CL_CHECK("Kernel (" + metaInfo.name + ") : Kernel Run",
                  clEnqueueNDRangeKernel(*((cl_command_queue*) dHandle->currentStream),
                                         kernel_,
                                         (cl_int) dims,
                                         NULL,
                                         (uintptr_t*) &fullOuter,
                                         (uintptr_t*) &inner,
                                         0, NULL, NULL));
  }

  template <>
  void kernel_t<OpenCL>::free(){
    OCCA_EXTRACT_DATA(OpenCL, Kernel);

    OCCA_CL_CHECK("Kernel: Free",
                  clReleaseKernel(data_.kernel));

    delete (OpenCLKernelData_t*) this->data;
  }
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<OpenCL>::memory_t(){
    strMode = "OpenCL";

    memInfo = memFlag::none;

    handle    = NULL;
    mappedPtr = NULL;
    uvaPtr    = NULL;

    dHandle = NULL;
    size = 0;

    textureInfo.dim = 1;
    textureInfo.w = textureInfo.h = textureInfo.d = 0;
  }

  template <>
  memory_t<OpenCL>::memory_t(const memory_t<OpenCL> &m){
    *this = m;
  }

  template <>
  memory_t<OpenCL>& memory_t<OpenCL>::operator = (const memory_t<OpenCL> &m){
    memInfo = m.memInfo;

    handle    = m.handle;
    mappedPtr = m.mappedPtr;
    uvaPtr    = m.uvaPtr;

    dHandle = m.dHandle;
    size    = m.size;

    textureInfo.dim = m.textureInfo.dim;

    textureInfo.w = m.textureInfo.w;
    textureInfo.h = m.textureInfo.h;
    textureInfo.d = m.textureInfo.d;

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
  void memory_t<OpenCL>::copyFrom(const void *src,
                                  const uintptr_t bytes,
                                  const uintptr_t offset){
    cl_command_queue &stream = *((cl_command_queue*) dHandle->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

    if(!isATexture())
      OCCA_CL_CHECK("Memory: Copy From",
                    clEnqueueWriteBuffer(stream, *((cl_mem*) handle),
                                         CL_TRUE,
                                         offset, bytes_, src,
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
                                        src,
                                        0, NULL, NULL));
    }
  }

  template <>
  void memory_t<OpenCL>::copyFrom(const memory_v *src,
                                  const uintptr_t bytes,
                                  const uintptr_t destOffset,
                                  const uintptr_t srcOffset){
    cl_command_queue &stream = *((cl_command_queue*) dHandle->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + destOffset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << destOffset << " , " << (destOffset + bytes_) << " ]");

    OCCA_CHECK((bytes_ + srcOffset) <= src->size,
               "Source has size [" << src->size << "],"
               << "trying to access [ " << srcOffset << " , " << (srcOffset + bytes_) << " ]");

    if(!isATexture())
      OCCA_CL_CHECK("Memory: Copy From",
                    clEnqueueCopyBuffer(stream,
                                        *((cl_mem*) src->handle),
                                        *((cl_mem*) handle),
                                        srcOffset, destOffset,
                                        bytes_,
                                        0, NULL, NULL));
    else
      OCCA_CL_CHECK("Texture Memory: Copy From",
                    clEnqueueCopyBuffer(stream,
                                        *((cl_mem*) src->handle),
                                        *((cl_mem*) handle),
                                        srcOffset, destOffset,
                                        bytes_,
                                        0, NULL, NULL));
  }

  template <>
  void memory_t<OpenCL>::copyTo(void *dest,
                                const uintptr_t bytes,
                                const uintptr_t offset){
    const cl_command_queue &stream = *((cl_command_queue*) dHandle->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

    if(!isATexture())
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
    const cl_command_queue &stream = *((cl_command_queue*) dHandle->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + srcOffset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << srcOffset << " , " << (srcOffset + bytes_) << " ]");

    OCCA_CHECK((bytes_ + destOffset) <= dest->size,
               "Destination has size [" << dest->size << "],"
               << "trying to access [ " << destOffset << " , " << (destOffset + bytes_) << " ]");

    if(!isATexture())
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
  void memory_t<OpenCL>::asyncCopyFrom(const void *src,
                                       const uintptr_t bytes,
                                       const uintptr_t offset){
    const cl_command_queue &stream = *((cl_command_queue*) dHandle->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

    if(!isATexture())
      OCCA_CL_CHECK("Memory: Asynchronous Copy From",
                    clEnqueueWriteBuffer(stream, *((cl_mem*) handle),
                                         CL_FALSE,
                                         offset, bytes_, src,
                                         0, NULL, NULL));
    else
      OCCA_CL_CHECK("Texture Memory: Asynchronous Copy From",
                    clEnqueueWriteBuffer(stream, *((cl_mem*) handle),
                                         CL_FALSE,
                                         offset, bytes_, src,
                                         0, NULL, NULL));
  }

  template <>
  void memory_t<OpenCL>::asyncCopyFrom(const memory_v *src,
                                       const uintptr_t bytes,
                                       const uintptr_t destOffset,
                                       const uintptr_t srcOffset){
    const cl_command_queue &stream = *((cl_command_queue*) dHandle->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + destOffset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << destOffset << " , " << (destOffset + bytes_) << " ]");

    OCCA_CHECK((bytes_ + srcOffset) <= src->size,
               "Source has size [" << src->size << "],"
               << "trying to access [ " << srcOffset << " , " << (srcOffset + bytes_) << " ]");

    if(!isATexture())
      OCCA_CL_CHECK("Memory: Asynchronous Copy From",
                    clEnqueueCopyBuffer(stream,
                                        *((cl_mem*) src->handle),
                                        *((cl_mem*) handle),
                                        srcOffset, destOffset,
                                        bytes_,
                                        0, NULL, NULL));
    else
      OCCA_CL_CHECK("Texture Memory: Asynchronous Copy From",
                    clEnqueueCopyBuffer(stream,
                                        *((cl_mem*) src->handle),
                                        *((cl_mem*) handle),
                                        srcOffset, destOffset,
                                        bytes_,
                                        0, NULL, NULL));
  }

  template <>
  void memory_t<OpenCL>::asyncCopyTo(void *dest,
                                     const uintptr_t bytes,
                                     const uintptr_t offset){
    const cl_command_queue &stream = *((cl_command_queue*) dHandle->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

    if(!isATexture())
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
    const cl_command_queue &stream = *((cl_command_queue*) dHandle->currentStream);

    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + srcOffset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << srcOffset << " , " << (srcOffset + bytes_) << " ]");

    OCCA_CHECK((bytes_ + destOffset) <= dest->size,
               "Destination has size [" << dest->size << "],"
               << "trying to access [ " << destOffset << " , " << (destOffset + bytes_) << " ]");

    if(!isATexture())
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
  void memory_t<OpenCL>::mappedFree(){
    cl_command_queue &stream = *((cl_command_queue*) dHandle->currentStream);
    cl_int error;

    // Un-map pointer
    error = clEnqueueUnmapMemObject(stream,
                                    *((cl_mem*) handle),
                                    mappedPtr,
                                    0, NULL, NULL);

    OCCA_CL_CHECK("Mapped Free: clEnqueueUnmapMemObject", error);

    // Free mapped-host pointer
    error = clReleaseMemObject(*((cl_mem*) handle));

    OCCA_CL_CHECK("Mapped Free: clReleaseMemObject", error);

    delete (cl_mem*) handle;
  }

  template <>
  void memory_t<OpenCL>::free(){
    clReleaseMemObject(*((cl_mem*) handle));

    if(!isAWrapper())
      delete (cl_mem*) handle;

    if(isATexture()){
      clReleaseSampler( *((cl_sampler*) textureInfo.arg) );

      if(!isAWrapper())
        delete (cl_sampler*) textureInfo.arg;
    }

    size = 0;
  }
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<OpenCL>::device_t() {
    strMode = "OpenCL";

    data = NULL;

    uvaEnabled_ = false;

    bytesAllocated = 0;

    getEnvironmentVariables();
  }

  template <>
  device_t<OpenCL>::device_t(const device_t<OpenCL> &d){
    *this = d;
  }

  template <>
  device_t<OpenCL>& device_t<OpenCL>::operator = (const device_t<OpenCL> &d){
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
  void* device_t<OpenCL>::getContextHandle(){
    OCCA_EXTRACT_DATA(OpenCL, Device);

    return (void*) data_.context;
  }

  template <>
  void device_t<OpenCL>::setup(argInfoMap &aim){
    properties = aim;

    data = new OpenCLDeviceData_t;

    OCCA_EXTRACT_DATA(OpenCL, Device);
    cl_int error;

    OCCA_CHECK(aim.has("platformID"),
               "[OpenCL] device not given [platformID]");

    OCCA_CHECK(aim.has("deviceID"),
               "[OpenCL] device not given [deviceID]");

    data_.platform = aim.iGet("platformID");
    data_.device   = aim.iGet("deviceID");

    data_.platformID = cl::platformID(data_.platform);
    data_.deviceID   = cl::deviceID(data_.platform, data_.device);

    data_.context = clCreateContext(NULL, 1, &data_.deviceID, NULL, NULL, &error);
    OCCA_CL_CHECK("Device: Creating Context", error);
  }

  template <>
  void device_t<OpenCL>::addOccaHeadersToInfo(kernelInfo &info_){
    info_.mode = OpenCL;
  }

  template <>
  std::string device_t<OpenCL>::getInfoSalt(const kernelInfo &info_){
    OCCA_EXTRACT_DATA(OpenCL, Device);

    std::stringstream salt;

    salt << "OpenCL"
         << data_.platform << '-' << data_.device
         << info_.salt()
         << parserVersion
         << compilerEnvScript
         << compiler
         << compilerFlags;

    return salt.str();
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
    int platformCount = occa::cl::getPlatformCount();

    for(int p = 0; p < platformCount; ++p){
      int deviceCount = occa::cl::getDeviceCountInPlatform(p);

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
  void device_t<OpenCL>::flush(){
    OCCA_CL_CHECK("Device: Flush",
                  clFlush(*((cl_command_queue*) currentStream)));
  }

  template <>
  void device_t<OpenCL>::finish(){
    OCCA_CL_CHECK("Device: Finish",
                  clFinish(*((cl_command_queue*) currentStream)));
  }

  template <>
  bool device_t<OpenCL>::fakesUva(){
    return true;
  }

  template <>
  void device_t<OpenCL>::waitFor(streamTag tag){
    OCCA_CL_CHECK("Device: Waiting For Tag",
                  clWaitForEvents(1, &(tag.clEvent())));
  }

  template <>
  stream_t device_t<OpenCL>::createStream(){
    OCCA_EXTRACT_DATA(OpenCL, Device);
    cl_int error;

    cl_command_queue *retStream = new cl_command_queue;

    *retStream = clCreateCommandQueue(data_.context, data_.deviceID, CL_QUEUE_PROFILING_ENABLE, &error);
    OCCA_CL_CHECK("Device: createStream", error);

    return retStream;
  }

  template <>
  void device_t<OpenCL>::freeStream(stream_t s){
    OCCA_CL_CHECK("Device: freeStream",
                  clReleaseCommandQueue( *((cl_command_queue*) s) ));

    delete (cl_command_queue*) s;
  }

  template <>
  stream_t device_t<OpenCL>::wrapStream(void *handle_){
    return handle_;
  }

  template <>
  streamTag device_t<OpenCL>::tagStream(){
    cl_command_queue &stream = *((cl_command_queue*) currentStream);

    streamTag ret;

#ifdef CL_VERSION_1_2
    OCCA_CL_CHECK("Device: Tagging Stream",
                  clEnqueueMarkerWithWaitList(stream, 0, NULL, &(ret.clEvent())));
#else
    OCCA_CL_CHECK("Device: Tagging Stream",
                  clEnqueueMarker(stream, &(ret.clEvent())));
#endif

    return ret;
  }

  template <>
  double device_t<OpenCL>::timeBetween(const streamTag &startTag, const streamTag &endTag){
    cl_ulong start, end;

    finish();

    OCCA_CL_CHECK ("Device: Time Between Tags (Start)",
                   clGetEventProfilingInfo(startTag.clEvent(),
                                           CL_PROFILING_COMMAND_END,
                                           sizeof(cl_ulong),
                                           &start, NULL) );

    OCCA_CL_CHECK ("Device: Time Between Tags (End)",
                   clGetEventProfilingInfo(endTag.clEvent(),
                                           CL_PROFILING_COMMAND_START,
                                           sizeof(cl_ulong),
                                           &end, NULL) );

    OCCA_CL_CHECK("Device: Time Between Tags (Freeing start tag)",
                  clReleaseEvent(startTag.clEvent()));

    OCCA_CL_CHECK("Device: Time Between Tags (Freeing end tag)",
                  clReleaseEvent(endTag.clEvent()));

    return (double) (1.0e-9 * (double)(end - start));
  }

  template <>
  std::string device_t<OpenCL>::fixBinaryName(const std::string &filename){
    return filename;
  }

  template <>
  kernel_v* device_t<OpenCL>::buildKernelFromSource(const std::string &filename,
                                                    const std::string &functionName,
                                                    const kernelInfo &info_){
    OCCA_EXTRACT_DATA(OpenCL, Device);

    kernel_v *k = new kernel_t<OpenCL>;

    k->dHandle = this;
    k->data    = new OpenCLKernelData_t;

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

    k->dHandle = this;
    k->data    = new OpenCLKernelData_t;

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
#if 0
    //---[ Creating shared library ]----
    kernel tmpK = occa::device(this).buildKernelFromSource(filename, functionName, info_);
    tmpK.free();

    kernelInfo info = info_;

    addOccaHeadersToInfo(info);

    std::string cachedBinary = getCachedName(filename, getInfoSalt(info));

    std::string prefix, name;
    getFilePrefixAndName(cachedBinary, prefix, name);

    std::string extension = getFileExtension(filename);

    const std::string iCachedBinary = prefix + "i_" + name;

    std::string contents = readFile(iCachedBinary);
    //==================================

    library::infoID_t infoID;

    infoID.modelID    = modelID_;
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
#endif
  }

  template <>
  kernel_v* device_t<OpenCL>::loadKernelFromLibrary(const char *cache,
                                                    const std::string &functionName){
#if 0
    OCCA_EXTRACT_DATA(OpenCL, Device);

    kernel_v *k = new kernel_t<OpenCL>;

    k->dHandle = this;
    k->data    = new OpenCLKernelData_t;

    OpenCLKernelData_t &kData_ = *((OpenCLKernelData_t*) k->data);

    kData_.platform = data_.platform;
    kData_.device   = data_.device;

    kData_.platformID = data_.platformID;
    kData_.deviceID   = data_.deviceID;
    kData_.context    = data_.context;

    k->loadFromLibrary(cache, functionName);
    return k;
#endif

    return NULL;
  }

  template <>
  memory_v* device_t<OpenCL>::wrapMemory(void *handle_,
                                         const uintptr_t bytes){
    memory_v *mem = new memory_t<OpenCL>;

    mem->dHandle = this;
    mem->size    = bytes;
    mem->handle  = handle_;

    mem->memInfo |= memFlag::isAWrapper;

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

    mem->dHandle = this;
    mem->size    = (dims.x * dims.y) * type.bytes();
    mem->handle  = handle_;

    mem->memInfo |= (memFlag::isATexture |
                     memFlag::isAWrapper);

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

    mem->dHandle = this;
    mem->size    = ((dim == 1) ? dims.x : (dims.x * dims.y)) * type.bytes();
    mem->handle  = handle_;

    mem->memInfo |= (memFlag::isATexture |
                     memFlag::isAWrapper);

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
                                     void *src){
    OCCA_EXTRACT_DATA(OpenCL, Device);

    memory_v *mem = new memory_t<OpenCL>;
    cl_int error;

    mem->dHandle = this;
    mem->handle  = new cl_mem;
    mem->size    = bytes;

    if(src == NULL){
      *((cl_mem*) mem->handle) = clCreateBuffer(data_.context,
                                                CL_MEM_READ_WRITE,
                                                bytes, NULL, &error);
    }
    else{
      *((cl_mem*) mem->handle) = clCreateBuffer(data_.context,
                                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                bytes, src, &error);

      finish();
    }

    return mem;
  }

  template <>
  memory_v* device_t<OpenCL>::textureAlloc(const int dim, const occa::dim &dims,
                                           void *src,
                                           occa::formatType type, const int permissions){
#ifndef CL_VERSION_1_2
    if(dim == 1)
      return malloc(dims.x * type.bytes(), src);

    OCCA_EXTRACT_DATA(OpenCL, Device);

    memory_v *mem = new memory_t<OpenCL>;
    cl_int error;

    mem->dHandle = this;
    mem->handle  = new cl_mem;
    mem->size    = (dims.x * dims.y) * type.bytes();

    mem->memInfo |= memFlag::isATexture;

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
                                               src, &error);

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

    mem->dHandle = this;
    mem->handle  = new cl_mem;
    mem->size    = ((dim == 1) ? dims.x : (dims.x * dims.y)) * type.bytes();

    mem->memInfo |= memFlag::isATexture;

    mem->textureInfo.dim  = dim;

    mem->textureInfo.w = dims.x;
    mem->textureInfo.h = dims.y;
    mem->textureInfo.d = dims.z;

    mem->textureInfo.bytesInEntry = type.bytes();

    cl_mem_flags flag;
    cl_image_format imageFormat;
    cl_image_desc imageDesc;

    flag  = CL_MEM_COPY_HOST_PTR;
    flag |= ((permissions == occa::readWrite) ? CL_MEM_READ_WRITE : CL_MEM_READ_ONLY);

    const int count = type.count();

    switch(count){
    case 1: imageFormat.image_channel_order = CL_R;    break;
    case 2: imageFormat.image_channel_order = CL_RG;   break;
    case 4: imageFormat.image_channel_order = CL_RGBA; break;
    };

    imageFormat.image_channel_data_type = *((cl_channel_type*) type.format<OpenCL>());

    imageDesc.image_type        = (dim == 1) ? CL_MEM_OBJECT_IMAGE1D : CL_MEM_OBJECT_IMAGE2D;
    imageDesc.image_width       = dims.x;
    imageDesc.image_height      = (dim < 2) ? 0 : dims.y;
    imageDesc.image_depth       = (dim < 3) ? 0 : dims.z;
    imageDesc.image_array_size  = 1;
    imageDesc.image_row_pitch   = 0;
    imageDesc.image_slice_pitch = 0;
    imageDesc.num_mip_levels    = 0;
    imageDesc.num_samples       = 0;
    imageDesc.buffer            = NULL;

    //--------------------------------------------
    cl_uint imageFormatCount;
    cl_image_format imageFormats[1024];

    clGetSupportedImageFormats(data_.context,
                               flag,
                               imageDesc.image_type,
                               0, NULL,
                               &imageFormatCount);

    clGetSupportedImageFormats(data_.context,
                               flag,
                               imageDesc.image_type,
                               imageFormatCount, imageFormats,
                               NULL);

    bool isCompatible = cl::imageFormatIsSupported(imageFormat,
                                                   imageFormats,
                                                   imageFormatCount);

    OCCA_CHECK(isCompatible,
               "The specified image format is not compatible");
    //============================================

    *((cl_mem*) mem->handle) = clCreateImage(data_.context, flag,
                                             &imageFormat, &imageDesc,
                                             src, &error);

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
  memory_v* device_t<OpenCL>::mappedAlloc(const uintptr_t bytes,
                                          void *src){

    OCCA_EXTRACT_DATA(OpenCL, Device);

    cl_command_queue &stream = *((cl_command_queue*) currentStream);

    memory_v *mem = new memory_t<OpenCL>;
    cl_int error;

    mem->dHandle  = this;
    mem->handle   = new cl_mem;
    mem->size     = bytes;

    mem->memInfo |= memFlag::isMapped;

    // Alloc pinned host buffer
    *((cl_mem*) mem->handle) = clCreateBuffer(data_.context,
                                              CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                              bytes,
                                              NULL, &error);

    OCCA_CL_CHECK("Device: clCreateBuffer", error);

    if(src != NULL)
      mem->copyFrom(src);

    // Map memory to read/write
    mem->mappedPtr = clEnqueueMapBuffer(stream,
                                        *((cl_mem*) mem->handle),
                                        CL_TRUE,
                                        CL_MAP_READ | CL_MAP_WRITE,
                                        0, bytes,
                                        0, NULL, NULL,
                                        &error);

    OCCA_CL_CHECK("Device: clEnqueueMapBuffer", error);

    // Sync memory mapping
    finish();

    return mem;
  }

  template <>
  uintptr_t device_t<OpenCL>::memorySize(){
    OCCA_EXTRACT_DATA(OpenCL, Device);

    return cl::getDeviceMemorySize(data_.deviceID);
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
    bool isGPU = false;

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
      OCCA_CHECK(false, "Can only find SIMD width for CPU and GPU devices at the momement\n");
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
        OCCA_CHECK(false, "simdWidth() only available for AMD, Intel and NVIDIA architectures");
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
}

#endif
