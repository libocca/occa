#if OCCA_HSA_ENABLED

#include "occa/HSA.hpp"

namespace occa {
  //---[ Helper Functions ]-----------
  namespace hsa {
    bool isInitialized = false;

    void init(){
      if(isInitialized)
        return;

      // 1) make sure that HSA is initialized
      OCCA_HSA_CHECK("Initializing HSA",
                     hsa_init());

      isInitialized = true;
    }

    int getDeviceCount(){
      int deviceCount;
      //      OCCA_HSA_CHECK("Finding Number of Devices",
      //                      cuDeviceGetCount(&deviceCount));

      // HSA equivalent ?

      return deviceCount;
    }

    CUdevice getDevice(const int id){
      CUdevice device;
      OCCA_HSA_CHECK("Getting CUdevice",
                      cuDeviceGet(&device, id));
      return device;
    }

    uintptr_t getDeviceMemorySize(CUdevice device){
      size_t bytes;
      OCCA_HSA_CHECK("Finding available memory on device",
                      hsaDeviceTotalMem(&bytes, device));
      return bytes;
    }

    std::string getDeviceListInfo(){
      std::stringstream ss;
      
      hsa::init();
      int deviceCount = hsa::getDeviceCount();
      if(deviceCount == 0)
        return "";

      char deviceName[1024];
      OCCA_HSA_CHECK("Getting Device Name",
                      hsaDeviceGetName(deviceName, 1024, 0));

      uintptr_t bytes      = getDeviceMemorySize(getDevice(0));
      std::string bytesStr = stringifyBytes(bytes);

      // << "==============o=======================o==========================================\n";
      ss << "     HSA     |  Device ID            | 0 "                                  << '\n'
         << "              |  Device Name          | " << deviceName                      << '\n'
         << "              |  Memory               | " << bytesStr                        << '\n';

      for(int i = 1; i < deviceCount; ++i){
        bytes    = getDeviceMemorySize(getDevice(i));
        bytesStr = stringifyBytes(bytes);

        OCCA_HSA_CHECK("Getting Device Name",
                        hsaDeviceGetName(deviceName, 1024, i));

        ss << "              |-----------------------+------------------------------------------\n"
           << "              |  Device ID            | " << i                                << '\n'
           << "              |  Device Name          | " << deviceName                       << '\n'
           << "              |  Memory               | " << bytesStr                         << '\n';
      }

      return ss.str();
    }

  }
  
  // ??
  const CUarray_format hsaFormats[8] = {CU_AD_FORMAT_UNSIGNED_INT8,
					CU_AD_FORMAT_UNSIGNED_INT16,
					CU_AD_FORMAT_UNSIGNED_INT32,
					CU_AD_FORMAT_SIGNED_INT8,
					CU_AD_FORMAT_SIGNED_INT16,
					CU_AD_FORMAT_SIGNED_INT32,
					CU_AD_FORMAT_HALF,
					CU_AD_FORMAT_FLOAT};
  
  template <>
  void* formatType::format<occa::HSA>() const {
    return ((void*) &(hsaFormats[format_]));
  }
  
  const int HSA_ADDRESS_NONE  = 0; // hsaBoundaryModeNone
  const int HSA_ADDRESS_CLAMP = 1; // hsaBoundaryModeClamp

  //==================================
  
  //---[ Kernel ]---------------------
  template <>
  kernel_t<HSA>::kernel_t(){
    strMode = "HSA";

    data    = NULL;
    dHandle = NULL;

    dims  = 1;
    inner = occa::dim(1,1,1);
    outer = occa::dim(1,1,1);

    nestedKernelCount = 0;

    maximumInnerDimSize_ = 0;
    preferredDimSize_    = 0;
  }

  template <>
  kernel_t<HSA>::kernel_t(const kernel_t<HSA> &k){
    data    = k.data;
    dHandle = k.dHandle;

    metaInfo = k.metaInfo;

    dims  = k.dims;
    inner = k.inner;
    outer = k.outer;

    nestedKernelCount = k.nestedKernelCount;

    if(0 < nestedKernelCount){
      nestedKernels = new kernel[nestedKernelCount];

      for(int i = 0; i < nestedKernelCount; ++i)
        nestedKernels[i] = k.nestedKernels[i];
    }

    preferredDimSize_ = k.preferredDimSize_;
  }

  template <>
  kernel_t<HSA>& kernel_t<HSA>::operator = (const kernel_t<HSA> &k){
    data    = k.data;
    dHandle = k.dHandle;

    metaInfo = k.metaInfo;

    dims  = k.dims;
    inner = k.inner;
    outer = k.outer;

    nestedKernelCount = k.nestedKernelCount;

    if(0 < nestedKernelCount){
      nestedKernels = new kernel[nestedKernelCount];

      for(int i = 0; i < nestedKernelCount; ++i)
        nestedKernels[i] = k.nestedKernels[i];
    }

    return *this;
  }

  template <>
  kernel_t<HSA>::~kernel_t(){}

  template <>
  void* kernel_t<HSA>::getKernelHandle(){
    OCCA_EXTRACT_DATA(HSA, Kernel);

    return data_.function;
  }

  template <>
  void* kernel_t<HSA>::getProgramHandle(){
    OCCA_EXTRACT_DATA(HSA, Kernel);

    return data_.module;
  }

  template <>
  std::string kernel_t<HSA>::fixBinaryName(const std::string &filename){
    return filename;
  }

  template <>
  kernel_t<HSA>* kernel_t<HSA>::buildFromSource(const std::string &filename,
                                                  const std::string &functionName,
                                                  const kernelInfo &info_){

    name = functionName;

    // TW: what does this do ?
    OCCA_EXTRACT_DATA(HSA, Kernel);
    kernelInfo info = info_;

    // TW: prepend headers to the kernel source code file
    dHandle->addOccaHeadersToInfo(info);

    // TW: create hash for kernel source code file
    const std::string hash = getFileContentHash(filename,
                                                dHandle->getInfoSalt(info));

    const std::string hashDir       = hashDirFor(filename, hash);
    const std::string sourceFile    = hashDir + kc::sourceFile;
    const std::string binaryFile    = hashDir + fixBinaryName(kc::binaryFile);
    const std::string ptxBinaryFile = hashDir + "ptxBinary.o";
    
    // TW: check for hash
    if(!haveHash(hash, 0)){
      waitForHash(hash, 0);

      if(verboseCompilation_f)
        std::cout << "Found cached binary of [" << compressFilename(filename) << "] in [" << compressFilename(binaryFile) << "]\n";

      // TW: build kernel from binary
      return buildFromBinary(binaryFile, functionName);
    }

    if(sys::fileExists(binaryFile)){
      releaseHash(hash, 0);

      if(verboseCompilation_f)
        std::cout << "Found cached binary of [" << compressFilename(filename) << "] in [" << compressFilename(binaryFile) << "]\n";

      return buildFromBinary(binaryFile, functionName);
    }

    createSourceFileFrom(filename, hashDir, info);

    // TW: this autosets some compiler flags for runtime CUDA kernel compilation

    std::string archSM = "";

    if((dHandle->compilerFlags.find("-arch=sm_") == std::string::npos) &&
       (            info.flags.find("-arch=sm_") == std::string::npos)){

      std::stringstream archSM_;

      int major, minor;
      OCCA_HSA_CHECK("Kernel (" + functionName + ") : Getting HSA Device Arch",
                      cuDeviceComputeCapability(&major, &minor, data_.device) );

      archSM_ << " -arch=sm_" << major << minor << ' ';

      archSM = archSM_.str();
    }

    // TW: this specifies the system command for compilation of kernels
    std::stringstream command;

    if(verboseCompilation_f)
      std::cout << "Compiling [" << functionName << "]\n";

    //---[ Compiling Command ]----------
    command.str("");

    command << dHandle->compiler
            << " -o "       << binaryFile
            << " -ptx -I."
            << " -I"  << env::OCCA_DIR << "/include"
            << ' '          << dHandle->compilerFlags
            << archSM
            << ' '          << info.flags
            << " -x cu "    << sourceFile;

    const std::string &sCommand = command.str();

    if(verboseCompilation_f)
      std::cout << sCommand << '\n';

    // TW: this does the compilation step
    const int compileError = system(sCommand.c_str());

    if(compileError){
      releaseHash(hash, 0);
      OCCA_CHECK(false, "Compilation error");
    }

    const CUresult moduleLoadError = cuModuleLoad(&data_.module,
                                                  binaryFile.c_str());

    if(moduleLoadError)
      releaseHash(hash, 0);

    OCCA_HSA_CHECK("Kernel (" + functionName + ") : Loading Module",
                    moduleLoadError);

    const CUresult moduleGetFunctionError = cuModuleGetFunction(&data_.function,
                                                                data_.module,
                                                                functionName.c_str());

    if(moduleGetFunctionError)
      releaseHash(hash, 0);

    OCCA_HSA_CHECK("Kernel (" + functionName + ") : Loading Function",
                    moduleGetFunctionError);

    releaseHash(hash, 0);

    return this;
  }

  template <>
  kernel_t<HSA>* kernel_t<HSA>::buildFromBinary(const std::string &filename,
                                                 const std::string &functionName){

    name = functionName;

    OCCA_EXTRACT_DATA(HSA, Kernel);

    OCCA_HSA_CHECK("Kernel (" + functionName + ") : Loading Module",
                    cuModuleLoad(&data_.module, filename.c_str()));

    OCCA_HSA_CHECK("Kernel (" + functionName + ") : Loading Function",
                    cuModuleGetFunction(&data_.function, data_.module, functionName.c_str()));

    return this;
  }

  template <>
  kernel_t<HSA>* kernel_t<HSA>::loadFromLibrary(const char *cache,
                                                  const std::string &functionName){
    OCCA_EXTRACT_DATA(HSA, Kernel);

    OCCA_HSA_CHECK("Kernel (" + functionName + ") : Loading Module",
                    cuModuleLoadData(&data_.module, cache));

    OCCA_HSA_CHECK("Kernel (" + functionName + ") : Loading Function",
                    cuModuleGetFunction(&data_.function, data_.module, functionName.c_str()));

    return this;
  }

  template <>
  uintptr_t kernel_t<HSA>::maximumInnerDimSize(){
    if(maximumInnerDimSize_)
      return maximumInnerDimSize_;

    OCCA_EXTRACT_DATA(HSA, Kernel);

    int maxSize;

    OCCA_HSA_CHECK("Kernel: Getting Maximum Inner-Dim Size",
                    cuFuncGetAttribute(&maxSize, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, data_.function));

    maximumInnerDimSize_ = (uintptr_t) maxSize;

    return maximumInnerDimSize_;
  }

  template <>
  int kernel_t<HSA>::preferredDimSize(){
    preferredDimSize_ = 32;
    return 32;
  }

#include "operators/HSAKernelOperators.cpp"

  template <>
  void kernel_t<HSA>::free(){
    OCCA_EXTRACT_DATA(HSA, Kernel);

    OCCA_HSA_CHECK("Kernel (" + name + ") : Unloading Module",
                    cuModuleUnload(data_.module));

    delete (HSAKernelData_t*) this->data;
  }
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<HSA>::memory_t(){
    strMode = "HSA";

    memInfo = memFlag::none;

    handle    = NULL;
    mappedPtr = NULL;
    uvaPtr    = NULL;

    dHandle = NULL;
    size    = 0;

  }

  template <>
  memory_t<HSA>::memory_t(const memory_t<HSA> &m){
    *this = m;
  }

  template <>
  memory_t<HSA>& memory_t<HSA>::operator = (const memory_t<HSA> &m){
    memInfo = m.memInfo;

    handle    = m.handle;
    mappedPtr = m.mappedPtr;
    uvaPtr    = m.uvaPtr;

    dHandle = m.dHandle;
    size    = m.size;


    return *this;
  }

  template <>
  memory_t<HSA>::~memory_t(){}

  template <>
  void* memory_t<HSA>::getMemoryHandle(){
    return handle;
  }

  template <>
  void memory_t<HSA>::copyFrom(const void *src,
                                const uintptr_t bytes,
                                const uintptr_t offset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

    OCCA_HSA_CHECK("Memory: Copy From",
		   cuMemcpyHtoD(*((CUdeviceptr*) handle) + offset, src, bytes_) );
  }

  template <>
  void memory_t<HSA>::copyFrom(const memory_v *src,
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

    void *dstPtr, *srcPtr;

    dstPtr = handle;
    
    srcPtr = src->handle;


    OCCA_HSA_CHECK("Memory: Copy From [Memory -> Memory]",
		   cuMemcpyDtoD(*((CUdeviceptr*) dstPtr) + destOffset,
				*((CUdeviceptr*) srcPtr) + srcOffset,
				bytes_) );
  }

  template <>
  void memory_t<HSA>::copyTo(void *dest,
                              const uintptr_t bytes,
                              const uintptr_t offset){
    const uintptr_t bytes_ = (bytes == 0) ? size : bytes;

    OCCA_CHECK((bytes_ + offset) <= size,
               "Memory has size [" << size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

    OCCA_HSA_CHECK("Memory: Copy To",
		   cuMemcpyDtoH(dest, *((CUdeviceptr*) handle) + offset, bytes_) );
  }

  template <>
  void memory_t<HSA>::copyTo(memory_v *dest,
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

    void *dstPtr, *srcPtr;

    srcPtr = handle;

    dstPtr = dest->handle;
    
    OCCA_HSA_CHECK("Memory: Copy To [Memory -> Memory]",
		   cuMemcpyDtoD(*((CUdeviceptr*) dstPtr) + destOffset,
				*((CUdeviceptr*) srcPtr) + srcOffset,
				bytes_) );
  }

  template <>
  void memory_t<HSA>::mappedFree(){
    if(isMapped()){
      OCCA_HSA_CHECK("Device: mappedFree()",
                      cuMemFreeHost(mappedPtr));

      delete (CUdeviceptr*) handle;

      size = 0;
    }
  }

  template <>
  void memory_t<HSA>::free(){
    //    cuMemFree(*((CUdeviceptr*) handle));
    hsa_memory_free((HSAdeviceptr*) handle);
    
    if(!isAWrapper())
      delete (CUdeviceptr*) handle;

    size = 0;
  }
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<HSA>::device_t() {
    strMode = "HSA";

    data = NULL;

    uvaEnabled_ = false;

    bytesAllocated = 0;

    getEnvironmentVariables();
  }

  template <>
  device_t<HSA>::device_t(const device_t<HSA> &d){
    *this = d;
  }

  template <>
  device_t<HSA>& device_t<HSA>::operator = (const device_t<HSA> &d){
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
  void* device_t<HSA>::getContextHandle(){
    OCCA_EXTRACT_DATA(HSA, Device);

    return (void*) data_.context;
  }

  
  // http://www.hsafoundation.com/html/HSA_Library.htm#Runtime/Topics/01_Intro/initialization_and_agent_discovery.htm%3FTocPath%3DHSA%2520Runtime%2520Programmer%25E2%2580%2599s%2520Reference%2520Manual%2520Version%25201.0%2520%7CChapter%25201.%2520Introduction%7CProgramming%2520Model%7C_____1
  // call back for hsa_iterate_agent
  hsa_status_t get_kernel_agent(hsa_agent_t agent, void* data) { 
    
    // checks that requested agent has capability to dispatch kernel
    uint32_t features = 0;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_FEATURE, &features);
    if (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH) {
      // Store kernel agent in the application-provided buffer and return
      hsa_agent_t* ret = (hsa_agent_t*) data;
      *ret = agent;
      return HSA_STATUS_INFO_BREAK;
    }
    // Keep iterating
    return HSA_STATUS_SUCCESS;
  }


  // https://github.com/rollingthunder/hsa_mmul_sample/blob/master/main.c
  /*
   * Determines if a memory region can be used for kernarg
   * allocations.
   */
  static hsa_status_t get_kernarg_memory_region(hsa_region_t region, void* data) {
    hsa_region_segment_t segment;
    hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
    if (HSA_REGION_SEGMENT_GLOBAL != segment) {
      return HSA_STATUS_SUCCESS;
    }

    hsa_region_global_flag_t flags;
    hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
    if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
      hsa_region_t* ret = (hsa_region_t*) data;
      *ret = region;
      return HSA_STATUS_INFO_BREAK;
    }

    return HSA_STATUS_SUCCESS;
  }


  template <>
  void device_t<HSA>::setup(argInfoMap &aim){
    hsa::init();
    properties = aim;

    data = new HSADeviceData_t;

    OCCA_EXTRACT_DATA(HSA, Device);

    data_.p2pEnabled = false;

    OCCA_CHECK(aim.has("deviceID"),
               "[HSA] device not given [deviceID]");

    const int deviceID = aim.iGet("deviceID");

    // get agent (device)
    hsa_iterate_agents(get_kernel_agent, &kernel_agent);



  }

  template <>
  void device_t<HSA>::addOccaHeadersToInfo(kernelInfo &info_){
    info_.mode = HSA;
  }

  template <>
  std::string device_t<HSA>::getInfoSalt(const kernelInfo &info_){
    std::stringstream salt;

    salt << "HSA"
         << info_.salt()
         << parserVersion
         << compilerEnvScript
         << compiler
         << compilerFlags;

    return salt.str();
  }

  template <>
  deviceIdentifier device_t<HSA>::getIdentifier() const {
    deviceIdentifier dID;

    dID.mode_ = HSA;

    const size_t archPos = compilerFlags.find("-arch=sm_");

    if(archPos == std::string::npos){
      OCCA_EXTRACT_DATA(HSA, Device);

      std::stringstream archSM_;

      int major, minor;
      OCCA_HSA_CHECK("Getting HSA Device Arch",
                      cuDeviceComputeCapability(&major, &minor, data_.device) );

      archSM_ << major << minor;

      dID.flagMap["sm_arch"] = archSM_.str();
    }
    else{
      const char *c0 = (compilerFlags.c_str() + archPos);
      const char *c1 = c0;

      while((*c1 != '\0') && (*c1 != ' '))
        ++c1;

      dID.flagMap["sm_arch"] = std::string(c0, c1 - c0);
    }

    return dID;
  }

  template <>
  void device_t<HSA>::getEnvironmentVariables(){
    char *c_compiler = getenv("OCCA_HSA_COMPILER");

    if(c_compiler != NULL)
      compiler = std::string(c_compiler);
    else
      compiler = "nvcc";

    char *c_compilerFlags = getenv("OCCA_HSA_COMPILER_FLAGS");

    if(c_compilerFlags != NULL)
      compilerFlags = std::string(c_compilerFlags);
    else{
#if OCCA_DEBUG_ENABLED
      compilerFlags = "-g";
#else
      compilerFlags = "";
#endif
    }
  }

  template <>
  void device_t<HSA>::appendAvailableDevices(std::vector<device> &dList){
    hsa::init();

    int deviceCount = hsa::getDeviceCount();

    for(int i = 0; i < deviceCount; ++i){
      device d;
      d.setup("HSA", i, 0);

      dList.push_back(d);
    }
  }

  template <>
  void device_t<HSA>::setCompiler(const std::string &compiler_){
    compiler = compiler_;
  }

  template <>
  void device_t<HSA>::setCompilerEnvScript(const std::string &compilerEnvScript_){
    compilerEnvScript = compilerEnvScript_;
  }

  template <>
  void device_t<HSA>::setCompilerFlags(const std::string &compilerFlags_){
    compilerFlags = compilerFlags_;
  }

  template <>
  void device_t<HSA>::flush(){}

  template <>
  void device_t<HSA>::finish(){
    OCCA_HSA_CHECK("Device: Finish",
                    cuStreamSynchronize(*((CUstream*) currentStream)) );
  }

  template <>
  bool device_t<HSA>::fakesUva(){
    return true;
  }

  template <>
  void device_t<HSA>::waitFor(streamTag tag){
    OCCA_HSA_CHECK("Device: Waiting For Tag",
                    cuEventSynchronize(tag.cuEvent()));
  }

  template <>
  stream_t device_t<HSA>::createStream(){

    // treat hsa queue as a stream (mimics opencl)
    hsa_queue_t queue;

    // find maximum number of queue items
    uint32_t queue_size = 0;
    hsa_status_t err = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
    
    // need to add HSA_CHECK
    hsa_queue_create(kernel_agent, queue_size, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &queue);
    
    // find region on agent
    /*
     * Find a memory region that supports kernel arguments.
     */
    kernarg_region.handle=(uint64_t)-1;
    hsa_agent_iterate_regions(agent, get_kernarg_memory_region, &kernarg_region);
    err = (kernarg_region.handle == (uint64_t)-1) ? HSA_STATUS_ERROR : HSA_STATUS_SUCCESS;
    check(Finding a kernarg memory region, err);

    return &queue;
  }

  template <>
  void device_t<HSA>::freeStream(stream_t s){

    OCCA_HSA_CHECK("Device: freeStream", 
		   hsa_queue_destroy(kernel_agent));

    //     delete (hsa_queue) s; ??
  }

  template <>
  stream_t device_t<HSA>::wrapStream(void *handle_){
    return handle_;
  }

  template <>
  streamTag device_t<HSA>::tagStream(){
    streamTag ret;

    OCCA_HSA_CHECK("Device: Tagging Stream (Creating Tag)",
                    cuEventCreate(&(ret.cuEvent()), CU_EVENT_DEFAULT));

    OCCA_HSA_CHECK("Device: Tagging Stream",
                    cuEventRecord(ret.cuEvent(), 0));

    return ret;
  }

  template <>
  double device_t<HSA>::timeBetween(const streamTag &startTag, const streamTag &endTag){
    OCCA_HSA_CHECK("Device: Waiting for endTag",
                    cuEventSynchronize(endTag.cuEvent()));

    float msTimeTaken;
    OCCA_HSA_CHECK("Device: Timing Between Tags",
                    cuEventElapsedTime(&msTimeTaken, startTag.cuEvent(), endTag.cuEvent()));

    return (double) (1.0e-3 * (double) msTimeTaken);
  }

  template <>
  std::string device_t<HSA>::fixBinaryName(const std::string &filename){
    return filename;
  }

  template <>
  kernel_v* device_t<HSA>::buildKernelFromSource(const std::string &filename,
                                                 const std::string &functionName,
                                                 const kernelInfo &info_){
    OCCA_EXTRACT_DATA(HSA, Device);

    OCCA_HSA_CHECK("Device: Setting Context",
                    cuCtxSetCurrent(data_.context));

    kernel_v *k = new kernel_t<HSA>;

    k->dHandle = this;
    k->data    = new HSAKernelData_t;

    HSAKernelData_t &kData_ = *((HSAKernelData_t*) k->data);

    kData_.device  = data_.device;
    kData_.context = data_.context;

    k->buildFromSource(filename, functionName, info_);

    return k;
  }

  template <>
  kernel_v* device_t<HSA>::buildKernelFromBinary(const std::string &filename,
                                                 const std::string &functionName){
    OCCA_EXTRACT_DATA(HSA, Device);

    kernel_v *k = new kernel_t<HSA>;

    k->dHandle = this;
    k->data    = new HSAKernelData_t;

    HSAKernelData_t &kData_ = *((HSAKernelData_t*) k->data);

    kData_.device  = data_.device;
    kData_.context = data_.context;

    k->buildFromBinary(filename, functionName);
    return k;
  }

  template <>
  void device_t<HSA>::cacheKernelInLibrary(const std::string &filename,
                                            const std::string &functionName,
                                            const kernelInfo &info_){
#if 0
    //---[ Creating shared library ]----
    kernel tmpK = occa::device(this).buildKernelFromSource(filename, functionName, info_);
    tmpK.free();

    kernelInfo info = info_;

    addOccaHeadersToInfo(info);

    std::string cachedBinary = getCachedName(filename, getInfoSalt(info));
    std::string contents     = readFile(cachedBinary);
    //==================================

    library::infoID_t infoID;

    infoID.modelID    = modelID_;
    infoID.kernelName = functionName;

    library::infoHeader_t &header = library::headerMap[infoID];

    header.fileID = -1;
    header.mode   = HSA;

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
  kernel_v* device_t<HSA>::loadKernelFromLibrary(const char *cache,
                                                  const std::string &functionName){
#if 0
    OCCA_EXTRACT_DATA(HSA, Device);

    kernel_v *k = new kernel_t<HSA>;

    k->dHandle = this;
    k->data    = new HSAKernelData_t;

    HSAKernelData_t &kData_ = *((HSAKernelData_t*) k->data);

    kData_.device  = data_.device;
    kData_.context = data_.context;

    k->loadFromLibrary(cache, functionName);
    return k;
#endif
    return NULL;
  }

  template <>
  memory_v* device_t<HSA>::wrapMemory(void *handle_,
                                       const uintptr_t bytes){
    memory_v *mem = new memory_t<HSA>;

    mem->dHandle = this;
    mem->size    = bytes;
    mem->handle  = (CUdeviceptr*) handle_;

    mem->memInfo |= memFlag::isAWrapper;

    return mem;
  }


  template <>
  memory_v* device_t<HSA>::malloc(const uintptr_t bytes,
                                   void *src){
    OCCA_EXTRACT_DATA(HSA, Device);

    memory_v *mem = new memory_t<HSA>;

    mem->dHandle = this;
    mem->handle  = new HSAdeviceptr*;
    mem->size    = bytes;

    OCCA_HSA_CHECK("Device: malloc",
		   hsa_memory_allocate(kernarg_region, bytes, (HSAdeviceptr*) mem->handle));

    if(src != NULL)
      mem->copyFrom(src, bytes, 0);

    return mem;
  }

  // TW: what does this do ?
  template <>
  memory_v* device_t<HSA>::mappedAlloc(const uintptr_t bytes,
                                        void *src){
    OCCA_EXTRACT_DATA(HSA, Device);

    memory_v *mem = new memory_t<HSA>;

    mem->dHandle  = this;
    mem->handle   = new CUdeviceptr*;
    mem->size     = bytes;

    mem->memInfo |= memFlag::isMapped;

    OCCA_HSA_CHECK("Device: Setting Context",
                    cuCtxSetCurrent(data_.context));

    OCCA_HSA_CHECK("Device: malloc host",
                    cuMemAllocHost((void**) &(mem->mappedPtr), bytes));

    OCCA_HSA_CHECK("Device: get device pointer from host",
                    cuMemHostGetDevicePointer((CUdeviceptr*) mem->handle,
                                              mem->mappedPtr,
                                              0));

    if(src != NULL)
      ::memcpy(mem->mappedPtr, src, bytes);

    return mem;
  }

  template <>
  uintptr_t device_t<HSA>::memorySize(){
    OCCA_EXTRACT_DATA(HSA, Device);

    return hsa::getDeviceMemorySize(data_.device);
  }

  template <>
  void device_t<HSA>::free(){
    OCCA_EXTRACT_DATA(HSA, Device);

    OCCA_HSA_CHECK("Device: Freeing Context",
                    cuCtxDestroy(data_.context) );

    delete (HSADeviceData_t*) data;
  }

  // TW: how do/can we find the SIMD width for an HSA device
  template <>
  int device_t<HSA>::simdWidth(){
    if(simdWidth_)
      return simdWidth_;

    OCCA_EXTRACT_DATA(HSA, Device);

    OCCA_HSA_CHECK("Device: Get Warp Size",
                    cuDeviceGetAttribute(&simdWidth_,
                                         CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                                         data_.device) );

    return simdWidth_;
  }
  //==================================


  //---[ Error Handling ]-------------
  std::string hsaError(const CUresult errorCode){
    switch(errorCode){
    case HSA_STATUS_SUCCESS:                        return "HSA_STATUS_SUCCESS";
    case HSA_STATUS_INFO_BREAK:                     return "HSA_STATUS_INFO_BREAK";
    case HSA_STATUS_ERROR:                          return "HSA_STATUS_ERROR";
    case HSA_STATUS_ERROR_INVALID_ARGUMENT:         return "HSA_STATUS_ERROR_INVALID_ARGUMENT";
    case HSA_STATUS_ERROR_INVALID_QUEUE_CREATION:   return "HSA_STATUS_ERROR_INVALID_QUEUE_CREATION";
    case HSA_STATUS_ERROR_INVALID_ALLOCATION:       return "HSA_STATUS_ERROR_INVALID_ALLOCATION";
    case HSA_STATUS_ERROR_INVALID_AGENT:            return "HSA_STATUS_ERROR_INVALID_AGENT";
    case HSA_STATUS_ERROR_INVALID_REGION:           return "HSA_STATUS_ERROR_INVALID_REGION";
    case HSA_STATUS_ERROR_INVALID_SIGNAL:           return "HSA_STATUS_ERROR_INVALID_SIGNAL";
    case HSA_STATUS_ERROR_INVALID_QUEUE:            return "HSA_STATUS_ERROR_INVALID_QUEUE";
    case HSA_STATUS_ERROR_OUT_OF_RESOURCES:         return "HSA_STATUS_ERROR_OUT_OF_RESOURCES";
    case HSA_STATUS_ERROR_INVALID_PACKET_FORMAT:    return "HSA_STATUS_ERROR_INVALID_PACKET_FORMAT";
    case HSA_STATUS_ERROR_RESOURCE_FREE:            return "HSA_STATUS_ERROR_RESOURCE_FREE";
    case HSA_STATUS_ERROR_NOT_INITIALIZED:          return "HSA_STATUS_ERROR_NOT_INITIALIZED";
    case HSA_STATUS_ERROR_REFCOUNT_OVERFLOW:        return "HSA_STATUS_ERROR_REFCOUNT_OVERFLOW";
    case HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS:   return "HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS";
    case HSA_STATUS_ERROR_INVALID_INDEX:            return "HSA_STATUS_ERROR_INVALID_INDEX";
    case HSA_STATUS_ERROR_INVALID_ISA:              return "HSA_STATUS_ERROR_INVALID_ISA";
    case HSA_STATUS_ERROR_INVALID_ISA_NAME:         return "HSA_STATUS_ERROR_INVALID_ISA_NAME";
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT:      return "HSA_STATUS_ERROR_INVALID_CODE_OBJECT";
    case HSA_STATUS_ERROR_INVALID_EXECUTABLE:       return "HSA_STATUS_ERROR_INVALID_EXECUTABLE";
    case HSA_STATUS_ERROR_FROZEN_EXECUTABLE:        return "HSA_STATUS_ERROR_FROZEN_EXECUTABLE";
    case HSA_STATUS_ERROR_INVALID_SYMBOL_NAME:      return "HSA_STATUS_ERROR_INVALID_SYMBOL_NAME";
    case HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED: return "HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED";
    case HSA_STATUS_ERROR_VARIABLE_UNDEFINED:       return "HSA_STATUS_ERROR_VARIABLE_UNDEFINED";
    default:                                        return "UNKNOWN ERROR";
    };
    //==================================
  }
}

#endif
