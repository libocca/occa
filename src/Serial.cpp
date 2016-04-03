#include "occa/Serial.hpp"

namespace occa {
  //---[ Helper Functions ]-----------
  namespace cpu {
    std::string getFieldFrom(const std::string &command,
                             const std::string &field){
#if (OCCA_OS & LINUX)
      std::string shellToolsFile = sys::getFilename("[occa]/scripts/shellTools.sh");

      if(!sys::fileExists(shellToolsFile)){
        sys::mkpath(getFileDirectory(shellToolsFile));

        std::ofstream fs2;
        fs2.open(shellToolsFile.c_str());

        fs2 << getCachedScript("shellTools.sh");

        fs2.close();
      }

      std::stringstream ss;

      ss << "echo \"(. " << shellToolsFile << "; " << command << " '" << field << "')\" | bash";

      std::string sCommand = ss.str();

      FILE *fp;
      fp = popen(sCommand.c_str(), "r");

      const int bufferSize = 4096;
      char *buffer = new char[bufferSize];

      ignoreResult( fread(buffer, sizeof(char), bufferSize, fp) );

      pclose(fp);

      int end;

      for(end = 0; end < bufferSize; ++end){
        if(buffer[end] == '\n')
          break;
      }

      std::string ret(buffer, end);

      delete [] buffer;

      return ret;
#else
      return "";
#endif
    }

    std::string getProcessorName(){
#if   (OCCA_OS & LINUX_OS)
      return getFieldFrom("getCPUINFOField", "model name");
#elif (OCCA_OS == OSX_OS)
      size_t bufferSize = 100;
      char buffer[100];

      sysctlbyname("machdep.cpu.brand_string",
                   &buffer, &bufferSize,
                   NULL, 0);

      return std::string(buffer);
#elif (OCCA_OS == WINDOWS_OS)
      char buffer[MAX_COMPUTERNAME_LENGTH + 1];
      int bytes;

      GetComputerName((LPWSTR) buffer, (LPDWORD) &bytes);

      return std::string(buffer, bytes);
#endif
    }

    int getCoreCount(){
#if (OCCA_OS & (LINUX_OS | OSX_OS))
      return sysconf(_SC_NPROCESSORS_ONLN);
#elif (OCCA_OS == WINDOWS_OS)
      SYSTEM_INFO sysinfo;
      GetSystemInfo(&sysinfo);
      return sysinfo.dwNumberOfProcessors;
#endif
    }

    int getProcessorFrequency(){
#if   (OCCA_OS & LINUX_OS)
      std::stringstream ss;
      int freq;

      ss << getFieldFrom("getCPUINFOField", "cpu MHz");

      ss >> freq;

      return freq;
#elif (OCCA_OS == OSX_OS)
      uint64_t frequency = 0;
      size_t size = sizeof(frequency);

      int error = sysctlbyname("hw.cpufrequency", &frequency, &size, NULL, 0);

      OCCA_CHECK(error != ENOMEM,
                 "Error getting CPU Frequency.\n");

      return frequency/1.0e6;
#elif (OCCA_OS == WINDOWS_OS)
      LARGE_INTEGER performanceFrequency;
      QueryPerformanceFrequency(&performanceFrequency);

      return (int) (((double) performanceFrequency.QuadPart) / 1000.0);
#endif
    }

    std::string getProcessorCacheSize(int level){
#if   (OCCA_OS & LINUX_OS)
      std::stringstream field;

      field << 'L' << level;

      if(level == 1)
        field << 'd';

      field << " cache";

      return getFieldFrom("getLSCPUField", field.str());
#elif (OCCA_OS == OSX_OS)
      std::stringstream ss;
      ss << "hw.l" << level;

      if(level == 1)
        ss << 'd';

      ss << "cachesize";

      std::string field = ss.str();

      uint64_t cache = 0;
      size_t size = sizeof(cache);

      int error = sysctlbyname(field.c_str(), &cache, &size, NULL, 0);

      OCCA_CHECK(error != ENOMEM,
                 "Error getting L" << level << " Cache Size.\n");

      return stringifyBytes(cache);
#elif (OCCA_OS == WINDOWS_OS)
      std::stringstream ss;
      DWORD cache = 0;

      PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = NULL;
      int bytes;

      GetLogicalProcessorInformation(buffer, (LPDWORD) &bytes);

      OCCA_CHECK((GetLastError() == ERROR_INSUFFICIENT_BUFFER),
                 "[GetLogicalProcessorInformation] Failed");

      buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION) cpu::malloc(bytes);

      bool passed = GetLogicalProcessorInformation(buffer, (LPDWORD) &bytes);

      OCCA_CHECK(passed,
                 "[GetLogicalProcessorInformation] Failed");

      PSYSTEM_LOGICAL_PROCESSOR_INFORMATION pos = buffer;
      int off = 0;

      while((off + sizeof(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)) <= bytes){
        switch(pos->Relationship){
        case RelationCache:{
          CACHE_DESCRIPTOR info = pos->Cache;

          if (info.Level == level)
            cache = info.Size;
        }
        }

        if(cache)
          break;
      }

      cpu::free(buffer);

      return stringifyBytes(cache);
#endif
    }

    uintptr_t installedRAM(){
#if   (OCCA_OS & LINUX_OS)
      struct sysinfo info;

      const int error = sysinfo(&info);

      if(error != 0)
        return 0;

      return info.totalram;
#elif (OCCA_OS == OSX_OS)
      int64_t ram;

      int mib[2]   = {CTL_HW, HW_MEMSIZE};
      size_t bytes = sizeof(ram);

      sysctl(mib, 2, &ram, &bytes, NULL, 0);

      return ram;
#elif (OCCA_OS == WINDOWS_OS)
      return 0;
#endif
    }

    uintptr_t availableRAM(){
#if   (OCCA_OS & LINUX_OS)
      struct sysinfo info;

      const int error = sysinfo(&info);

      if(error != 0)
        return 0;

      return info.freeram;
#elif (OCCA_OS == OSX_OS)
      mach_msg_type_number_t infoCount = HOST_VM_INFO_COUNT;
      mach_port_t hostPort = mach_host_self();

      vm_statistics_data_t hostInfo;
      kern_return_t status;
      vm_size_t pageSize;

      status = host_page_size(hostPort, &pageSize);

      if(status != KERN_SUCCESS)
        return 0;

      status = host_statistics(hostPort,
                               HOST_VM_INFO,
                               (host_info_t) &hostInfo,
                               &infoCount);

      if(status != KERN_SUCCESS)
        return 0;

      return (hostInfo.free_count * pageSize);
#elif (OCCA_OS == WINDOWS_OS)
      return 0;
#endif
    }

    std::string getDeviceListInfo(){
      std::stringstream ss, ssFreq;

      ss << getCoreCount();

      uintptr_t ram      = installedRAM();
      std::string ramStr = stringifyBytes(ram);

      const int freq = getProcessorFrequency();

      if(freq < 1000)
        ssFreq << freq << " MHz";
      else
        ssFreq << (freq/1000.0) << " GHz";

      std::string l1 = getProcessorCacheSize(1);
      std::string l2 = getProcessorCacheSize(2);
      std::string l3 = getProcessorCacheSize(3);

      size_t maxSize = ((l1.size() < l2.size()) ? l2.size() : l1.size());
      maxSize        = ((maxSize   < l3.size()) ? l3.size() : maxSize  );

      if(maxSize){
        l1 = std::string(maxSize - l1.size(), ' ') + l1;
        l2 = std::string(maxSize - l2.size(), ' ') + l2;
        l3 = std::string(maxSize - l3.size(), ' ') + l3;
      }

      std::string tab[2];
      tab[0] = "   CPU Info   ";
      tab[1] = "              ";

      std::string processorName  = getProcessorName();
      std::string clockFrequency = ssFreq.str();
      std::string coreCount      = ss.str();

      ss.str("");

      // [P]rinted [S]omething
      bool ps = false;

      // << "==============o=======================o==========================================\n";
      if(processorName.size())
        ss << tab[ps]  << "|  Processor Name       | " << processorName                   << '\n'; ps = true;
      if(coreCount.size())
        ss << tab[ps]  << "|  Cores                | " << coreCount                       << '\n'; ps = true;
      if(ramStr.size())
        ss << tab[ps]  << "|  Memory (RAM)         | " << ramStr                          << '\n'; ps = true;
      if(clockFrequency.size())
        ss << tab[ps]  << "|  Clock Frequency      | " << clockFrequency                  << '\n'; ps = true;
      ss   << tab[ps]  << "|  SIMD Instruction Set | " << OCCA_VECTOR_SET                 << '\n'
           << tab[ps]  << "|  SIMD Width           | " << (32*OCCA_SIMD_WIDTH) << " bits" << '\n'; ps = true;
      if(l1.size())
        ss << tab[ps]  << "|  L1 Cache Size (d)    | " << l1                              << '\n'; ps = true;
      if(l2.size())
        ss << tab[ps]  << "|  L2 Cache Size        | " << l2                              << '\n'; ps = true;
      if(l3.size())
        ss << tab[ps]  << "|  L3 Cache Size        | " << l3                              << '\n';
      // << "==============o=======================o==========================================\n";

      return ss.str();
    }

    int compilerVendor(const std::string &compiler){
#if (OCCA_OS & (LINUX_OS | OSX_OS))
      std::stringstream ss;
      int vendor_ = cpu::vendor::notFound;

      const std::string safeCompiler = removeSlashes(compiler);
      const std::string &hash = safeCompiler;

      const std::string testFilename   = sys::getFilename("[occa]/testing/compilerVendorTest.cpp");
      const std::string binaryFilename = sys::getFilename("[occa]/testing/compilerVendor_" + safeCompiler);
      const std::string infoFilename   = sys::getFilename("[occa]/testing/compilerVendorInfo_" + safeCompiler);

      cacheFile(testFilename,
                readFile(env::OCCA_DIR + "/scripts/compilerVendorTest.cpp"),
                "compilerVendorTest");

      if(!haveHash(hash)){
        waitForHash(hash);
      } else {
        if(!sys::fileExists(infoFilename)){
          ss << compiler
             << ' '
             << testFilename
             << " -o "
             << binaryFilename
             << " > /dev/null 2>&1";

          const int compileError = system(ss.str().c_str());

          if(!compileError){
            int exitStatus = system(binaryFilename.c_str());
            int vendorBit  = WEXITSTATUS(exitStatus);

            if(vendorBit < cpu::vendor::b_max)
              vendor_ = (1 << vendorBit);
          }

          ss.str("");
          ss << vendor_;

          writeToFile(infoFilename, ss.str());
          releaseHash(hash);

          return vendor_;
        }
        releaseHash(hash);
      }

      ss << readFile(infoFilename);
      ss >> vendor_;

      return vendor_;

#elif (OCCA_OS == WINDOWS_OS)
#  if OCCA_USING_VS
      return cpu::vendor::VisualStudio;
#  endif

      if(compiler.find("cl.exe") != std::string::npos){
        return cpu::vendor::VisualStudio;
      }
#endif
    }

    std::string compilerSharedBinaryFlags(const std::string &compiler){
      return compilerSharedBinaryFlags( cpu::compilerVendor(compiler) );
    }

    std::string compilerSharedBinaryFlags(const int vendor_){
      if(vendor_ & (cpu::vendor::GNU   |
                    cpu::vendor::LLVM  |
                    cpu::vendor::Intel |
                    cpu::vendor::IBM   |
                    cpu::vendor::PGI   |
                    cpu::vendor::Cray  |
                    cpu::vendor::Pathscale)){

        return "-x c++ -fPIC -shared"; // [-] -x c++ for now
      }
      else if(vendor_ & cpu::vendor::HP){
        return "+z -b";
      }
      else if(vendor_ & cpu::vendor::VisualStudio){
#if OCCA_DEBUG_ENABLED
        return "/TP /LD /MDd";
#else
        return "/TP /LD /MD";
#endif
      }

      return "";
    }

    void addSharedBinaryFlagsTo(const std::string &compiler, std::string &flags){
      addSharedBinaryFlagsTo(cpu::compilerVendor(compiler), flags);
    }

    void addSharedBinaryFlagsTo(const int vendor_, std::string &flags){
      std::string sFlags = cpu::compilerSharedBinaryFlags(vendor_);

      if(flags.size() == 0)
        flags = sFlags;

      if(flags.find(sFlags) == std::string::npos)
        flags = (sFlags + " " + flags);
    }

    void* malloc(uintptr_t bytes){
      void* ptr;

#if   (OCCA_OS & (LINUX_OS | OSX_OS))
      ignoreResult( posix_memalign(&ptr, env::OCCA_MEM_BYTE_ALIGN, bytes) );
#elif (OCCA_OS == WINDOWS_OS)
      ptr = ::malloc(bytes);
#endif

      return ptr;
    }

    void free(void *ptr){
      ::free(ptr);
    }

    void* dlopen(const std::string &filename,
                 const std::string &hash){

#if (OCCA_OS & (LINUX_OS | OSX_OS))
      void *dlHandle = ::dlopen(filename.c_str(), RTLD_NOW);

      if((dlHandle == NULL) && (0 < hash.size())){
        releaseHash(hash, 0);

        OCCA_CHECK(false,
                   "Error loading binary [" << compressFilename(filename) << "] with dlopen");
      }
#else
      void *dlHandle = LoadLibraryA(filename.c_str());

      if((dlHandle == NULL) && (0 < hash.size())){
        releaseHash(hash, 0);

        OCCA_CHECK(dlHandle != NULL,
                   "Error loading dll [" << compressFilename(filename) << "] (WIN32 error: " << GetLastError() << ")");
      }
#endif

      return dlHandle;
    }

    handleFunction_t dlsym(void *dlHandle,
                           const std::string &functionName,
                           const std::string &hash){

#if (OCCA_OS & (LINUX_OS | OSX_OS))
      void *sym = ::dlsym(dlHandle, functionName.c_str());

      char *dlError;

      if(((dlError = dlerror()) != NULL) && (0 < hash.size())){
        releaseHash(hash, 0);

        OCCA_CHECK(false,
                   "Error loading symbol from binary with dlsym (DL Error: " << dlError << ")");
      }
#else
      void *sym = GetProcAddress((HMODULE) dlHandle, functionName.c_str());

      if((sym == NULL) && (0 < hash.size())){

        OCCA_CHECK(false,
                   "Error loading symbol from binary with GetProcAddress");
      }
#endif

      handleFunction_t sym2;

      ::memcpy(&sym2, &sym, sizeof(sym));

      return sym2;
    }

    void runFunction(handleFunction_t f,
                     const int *occaKernelInfoArgs,
                     int occaInnerId0, int occaInnerId1, int occaInnerId2,
                     int argc, void **args){

#include "operators/runFunctionFromArguments.cpp"
    }
  }
  //==================================


  //---[ Kernel ]---------------------
  template <>
  kernel_t<Serial>::kernel_t(){
    strMode = "Serial";

    data    = NULL;
    dHandle = NULL;

    dims  = 1;
    inner = occa::dim(1,1,1);
    outer = occa::dim(1,1,1);
  }

  template <>
  kernel_t<Serial>::kernel_t(const kernel_t<Serial> &k){
    *this = k;
  }

  template <>
  kernel_t<Serial>& kernel_t<Serial>::operator = (const kernel_t<Serial> &k){
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
  kernel_t<Serial>::~kernel_t(){}

  template <>
  void* kernel_t<Serial>::getKernelHandle(){
    OCCA_EXTRACT_DATA(Serial, Kernel);

    void *ret;

    ::memcpy(&ret, &data_.handle, sizeof(void*));

    return ret;
  }

  template <>
  void* kernel_t<Serial>::getProgramHandle(){
    OCCA_EXTRACT_DATA(Serial, Kernel);

    return data_.dlHandle;
  }

  template <>
  std::string kernel_t<Serial>::fixBinaryName(const std::string &filename){
#if (OCCA_OS & (LINUX_OS | OSX_OS))
    return filename;
#else
    return (filename + ".dll");
#endif
  }

  template <>
  kernel_t<Serial>* kernel_t<Serial>::buildFromSource(const std::string &filename,
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

    data = new SerialKernelData_t;

    createSourceFileFrom(filename, hashDir, info);

    std::stringstream command;

    if(dHandle->compilerEnvScript.size())
      command << dHandle->compilerEnvScript << " && ";

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

    OCCA_EXTRACT_DATA(Serial, Kernel);

    data_.dlHandle = cpu::dlopen(binaryFile, hash);
    data_.handle   = cpu::dlsym(data_.dlHandle, functionName, hash);

    releaseHash(hash, 0);

    return this;
  }

  template <>
  kernel_t<Serial>* kernel_t<Serial>::buildFromBinary(const std::string &filename,
                                                      const std::string &functionName){

    name = functionName;

    data = new SerialKernelData_t;

    OCCA_EXTRACT_DATA(Serial, Kernel);

    data_.dlHandle = cpu::dlopen(filename);
    data_.handle   = cpu::dlsym(data_.dlHandle, functionName);

    return this;
  }

  template <>
  kernel_t<Serial>* kernel_t<Serial>::loadFromLibrary(const char *cache,
                                                      const std::string &functionName){
    name = functionName;

    return buildFromBinary(cache, functionName);
  }

  template <>
  uintptr_t kernel_t<Serial>::maximumInnerDimSize(){
    return ((uintptr_t) -1);
  }

  // [-] Missing
  template <>
  int kernel_t<Serial>::preferredDimSize(){
    preferredDimSize_ = OCCA_SIMD_WIDTH;
    return OCCA_SIMD_WIDTH;
  }

  template <>
  void kernel_t<Serial>::runFromArguments(const int kArgc, const kernelArg *kArgs){
    SerialKernelData_t &data_ = *((SerialKernelData_t*) data);
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
  void kernel_t<Serial>::free(){
    OCCA_EXTRACT_DATA(Serial, Kernel);

#if (OCCA_OS & (LINUX_OS | OSX_OS))
    dlclose(data_.dlHandle);
#else
    FreeLibrary((HMODULE) (data_.dlHandle));
#endif
  }
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<Serial>::memory_t(){
    strMode = "Serial";

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
  memory_t<Serial>::memory_t(const memory_t<Serial> &m){
    *this = m;
  }

  template <>
  memory_t<Serial>& memory_t<Serial>::operator = (const memory_t<Serial> &m){
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
  memory_t<Serial>::~memory_t(){}

  template <>
  void* memory_t<Serial>::getMemoryHandle(){
    return handle;
  }

  template <>
  void* memory_t<Serial>::getTextureHandle(){
    return textureInfo.arg;
  }

  template <>
  void memory_t<Serial>::copyFrom(const void *src,
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
  void memory_t<Serial>::copyFrom(const memory_v *src,
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
  void memory_t<Serial>::copyTo(void *dest,
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
  void memory_t<Serial>::copyTo(memory_v *dest,
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
    const void *srcPtr = ((char*) (isATexture()       ? textureInfo.arg       : handle))       + srcOffset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<Serial>::asyncCopyFrom(const void *src,
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
  void memory_t<Serial>::asyncCopyFrom(const memory_v *src,
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

    void *destPtr      = ((char*) (isATexture()      ? textureInfo.arg      : handle))         + destOffset;
    const void *srcPtr = ((char*) (src->isATexture() ? src->textureInfo.arg : src->handle)) + srcOffset;;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<Serial>::asyncCopyTo(void *dest,
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
  void memory_t<Serial>::asyncCopyTo(memory_v *dest,
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
    const void *srcPtr = ((char*) (isATexture()       ? textureInfo.arg       : handle))       + srcOffset;

    ::memcpy(destPtr, srcPtr, bytes_);
  }

  template <>
  void memory_t<Serial>::mappedFree(){
    cpu::free(handle);
    handle    = NULL;
    mappedPtr = NULL;

    size = 0;
  }

  template <>
  void memory_t<Serial>::free(){
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
  device_t<Serial>::device_t(){
    strMode = "Serial";

    data = NULL;
    uvaEnabled_ = false;
    bytesAllocated = 0;

    getEnvironmentVariables();

    cpu::addSharedBinaryFlagsTo(compiler, compilerFlags);
  }

  template <>
  device_t<Serial>::device_t(const device_t<Serial> &d){
    *this = d;
  }

  template <>
  device_t<Serial>& device_t<Serial>::operator = (const device_t<Serial> &d){
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
  void* device_t<Serial>::getContextHandle(){
    return NULL;
  }

  template <>
  void device_t<Serial>::setup(argInfoMap &aim){
    properties = aim;

    data = new SerialDeviceData_t;

    OCCA_EXTRACT_DATA(Serial, Device);

    data_.vendor = cpu::compilerVendor(compiler);

    cpu::addSharedBinaryFlagsTo(data_.vendor, compilerFlags);
  }

  template <>
  void device_t<Serial>::addOccaHeadersToInfo(kernelInfo &info_){
    info_.mode = Serial;
  }

  template <>
  std::string device_t<Serial>::getInfoSalt(const kernelInfo &info_){
    std::stringstream salt;

    salt << "Serial"
         << info_.salt()
         << parserVersion
         << compilerEnvScript
         << compiler
         << compilerFlags;

    return salt.str();
  }

  template <>
  deviceIdentifier device_t<Serial>::getIdentifier() const {
    deviceIdentifier dID;

    dID.mode_ = Serial;

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
  void device_t<Serial>::getEnvironmentVariables(){
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
    compilerFlags = " /Od";
#  else
    compilerFlags = " /O2";
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
  void device_t<Serial>::appendAvailableDevices(std::vector<device> &dList){
    device d;
    d.setup("Serial");

    dList.push_back(d);
  }

  template <>
  void device_t<Serial>::setCompiler(const std::string &compiler_){
    compiler = compiler_;

    OCCA_EXTRACT_DATA(Serial, Device);

    data_.vendor = cpu::compilerVendor(compiler);

    cpu::addSharedBinaryFlagsTo(data_.vendor, compilerFlags);
  }

  template <>
  void device_t<Serial>::setCompilerEnvScript(const std::string &compilerEnvScript_){
    compilerEnvScript = compilerEnvScript_;
  }

  template <>
  void device_t<Serial>::setCompilerFlags(const std::string &compilerFlags_){
    OCCA_EXTRACT_DATA(Serial, Device);

    compilerFlags = compilerFlags_;

    cpu::addSharedBinaryFlagsTo(data_.vendor, compilerFlags);
  }

  template <>
  void device_t<Serial>::flush(){}

  template <>
  void device_t<Serial>::finish(){}

  template <>
  bool device_t<Serial>::fakesUva(){
    return false;
  }

  template <>
  void device_t<Serial>::waitFor(streamTag tag){}

  template <>
  stream_t device_t<Serial>::createStream(){
    return NULL;
  }

  template <>
  void device_t<Serial>::freeStream(stream_t s){}

  template <>
  stream_t device_t<Serial>::wrapStream(void *handle_){
    return NULL;
  }

  template <>
  streamTag device_t<Serial>::tagStream(){
    streamTag ret;

    ret.tagTime = currentTime();

    return ret;
  }

  template <>
  double device_t<Serial>::timeBetween(const streamTag &startTag, const streamTag &endTag){
    return (endTag.tagTime - startTag.tagTime);
  }

  template <>
  std::string device_t<Serial>::fixBinaryName(const std::string &filename){
#if (OCCA_OS & (LINUX_OS | OSX_OS))
    return filename;
#else
    return (filename + ".dll");
#endif
  }

  template <>
  kernel_v* device_t<Serial>::buildKernelFromSource(const std::string &filename,
                                                    const std::string &functionName,
                                                    const kernelInfo &info_){
    kernel_v *k = new kernel_t<Serial>;
    k->dHandle = this;

    k->buildFromSource(filename, functionName, info_);

    return k;
  }

  template <>
  kernel_v* device_t<Serial>::buildKernelFromBinary(const std::string &filename,
                                                    const std::string &functionName){
    kernel_v *k = new kernel_t<Serial>;
    k->dHandle = this;
    k->buildFromBinary(filename, functionName);
    return k;
  }

  template <>
  void device_t<Serial>::cacheKernelInLibrary(const std::string &filename,
                                              const std::string &functionName,
                                              const kernelInfo &info_){
#if 0
    //---[ Creating shared library ]----
    kernel tmpK = occa::device(this).buildKernelFromSource(filename, functionName, info_);
    tmpK.free();

    kernelInfo info = info_;

    addOccaHeadersToInfo(info);

    std::string cachedBinary = getCachedName(filename, getInfoSalt(info));

#if (OCCA_OS & WINDOWS_OS)
    // Windows requires .dll extension
    cachedBinary = cachedBinary + ".dll";
#endif
    //==================================

    library::infoID_t infoID;

    infoID.modelID    = modelID_;
    infoID.kernelName = functionName;

    library::infoHeader_t &header = library::headerMap[infoID];

    header.fileID = -1;
    header.mode   = Serial;

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
  kernel_v* device_t<Serial>::loadKernelFromLibrary(const char *cache,
                                                    const std::string &functionName){
#if 0
    kernel_v *k = new kernel_t<Serial>;
    k->dHandle = this;
    k->loadFromLibrary(cache, functionName);
    return k;
#endif
    return NULL;
  }

  template <>
  memory_v* device_t<Serial>::wrapMemory(void *handle_,
                                         const uintptr_t bytes){
    memory_v *mem = new memory_t<Serial>;

    mem->dHandle = this;
    mem->size    = bytes;
    mem->handle  = handle_;

    mem->memInfo |= memFlag::isAWrapper;

    return mem;
  }

  template <>
  memory_v* device_t<Serial>::wrapTexture(void *handle_,
                                          const int dim, const occa::dim &dims,
                                          occa::formatType type, const int permissions){
    memory_v *mem = new memory_t<Serial>;

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
  memory_v* device_t<Serial>::malloc(const uintptr_t bytes,
                                     void *src){
    memory_v *mem = new memory_t<Serial>;

    mem->dHandle = this;
    mem->size    = bytes;

    mem->handle = cpu::malloc(bytes);

    if(src != NULL)
      ::memcpy(mem->handle, src, bytes);

    return mem;
  }

  template <>
  memory_v* device_t<Serial>::textureAlloc(const int dim, const occa::dim &dims,
                                           void *src,
                                           occa::formatType type, const int permissions){
    memory_v *mem = new memory_t<Serial>;

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
  memory_v* device_t<Serial>::mappedAlloc(const uintptr_t bytes,
                                          void *src){
    memory_v *mem = malloc(bytes, src);

    mem->mappedPtr = mem->handle;

    return mem;
  }

  template <>
  uintptr_t device_t<Serial>::memorySize(){
    return cpu::installedRAM();
  }

  template <>
  void device_t<Serial>::free(){
    if(data){
      delete (SerialDeviceData_t*) data;
      data = NULL;
    }
  }

  template <>
  int device_t<Serial>::simdWidth(){
    simdWidth_ = OCCA_SIMD_WIDTH;
    return OCCA_SIMD_WIDTH;
  }
  //==================================
}
