#include "occa.hpp"

// Use events for timing!

namespace occa {
  //---[ Helper Classes ]-------------
  kernelInfo defaultKernelInfo;

  const char* deviceInfo::header = "| Name                                      | Num | Available Modes                  |";
  const char* deviceInfo::sLine  = "+-------------------------------------------+-----+----------------------------------+";

  const char* deviceInfo::dLine1 = "+--------------------------------------------------------+";
  const char* deviceInfo::dLine2 = "+  -  -  -  -  -  +  -  -  -  -  -  -  -  -  -  -  -  -  +";
  //==================================

  //---[ Kernel ]---------------------
  kernel::kernel() :
    mode_(),
    strMode(""),
    kHandle(NULL) {}

  kernel::kernel(const kernel &k) :
    mode_(k.mode_),
    strMode(k.strMode),
    kHandle(k.kHandle) {}

  kernel& kernel::operator = (const kernel &k){
    mode_   = k.mode_;
    strMode = k.strMode;

    kHandle = k.kHandle;

    return *this;
  }

  std::string& kernel::mode(){
    return strMode;
  }

  kernel& kernel::buildFromSource(const std::string &filename,
                                  const std::string &functionName_,
                                  const kernelInfo &info_){
    kHandle->buildFromSource(filename, functionName_, info_);

    return *this;
  }

  kernel& kernel::buildFromBinary(const std::string &filename,
                                  const std::string &functionName_){
    kHandle->buildFromBinary(filename, functionName_);

    return *this;
  }

  void kernel::setWorkingDims(int dims, occa::dim inner, occa::dim outer){
    for(int i = 0; i < dims; ++i){
      inner[i] += (inner[i] ? 0 : 1);
      outer[i] += (outer[i] ? 0 : 1);
    }

    for(int i = dims; i < 3; ++i)
      inner[i] = outer[i] = 1;

    kHandle->dims  = dims;
    kHandle->inner = inner;
    kHandle->outer = outer;
  }

  int kernel::preferredDimSize(){
    return kHandle->preferredDimSize();
  }

  void kernel::clearArgumentList(){
    argumentCount = 0;
  }

  void kernel::addArgument(const int argPos,
                           const kernelArg &arg){
    if(argumentCount < (argPos + 1)){
      OCCA_CHECK(argPos < OCCA_MAX_ARGS);

      argumentCount = (argPos + 1);
    }

    arguments[argPos] = arg;
  }

  void kernel::runFromArguments(){
    // [-] OCCA_MAX_ARGS = 25
#include "operators/occaRunFromArguments.cpp"

    return;
  }

#include "operators/occaOperatorDefinitions.cpp"

  double kernel::timeTaken(){
    return kHandle->timeTaken();
  }

  void kernel::free(){
    kHandle->free();
    delete kHandle;
  }
  //==================================


  //---[ Memory ]---------------------
  memory::memory() :
    mode_(),
    strMode(""),
    mHandle(NULL) {}

  memory::memory(const memory &m) :
    mode_(m.mode_),
    strMode(m.strMode),
    mHandle(m.mHandle) {}

  memory& memory::operator = (const memory &m){
    mode_   = m.mode_;
    strMode = m.strMode;

    mHandle = m.mHandle;

    return *this;
  }

  std::string& memory::mode(){
    return strMode;
  }

  void memory::copyFrom(const void *source,
                        const uintptr_t bytes,
                        const uintptr_t offset){
    mHandle->copyFrom(source, bytes, offset);
  }

  void memory::copyFrom(const memory &source,
                        const uintptr_t bytes,
                        const uintptr_t destOffset,
                        const uintptr_t srcOffset){
    mHandle->copyFrom(source.mHandle, bytes, destOffset, srcOffset);
  }

  void memory::copyTo(void *dest,
                      const uintptr_t bytes,
                      const uintptr_t offset){
    mHandle->copyTo(dest, bytes, offset);
  }

  void memory::copyTo(memory &dest,
                      const uintptr_t bytes,
                      const uintptr_t destOffset,
                      const uintptr_t srcOffset){
    mHandle->copyTo(dest.mHandle, bytes, destOffset, srcOffset);
  }

  void memory::asyncCopyFrom(const void *source,
                             const uintptr_t bytes,
                             const uintptr_t offset){
    mHandle->asyncCopyFrom(source, bytes, offset);
  }

  void memory::asyncCopyFrom(const memory &source,
                             const uintptr_t bytes,
                             const uintptr_t destOffset,
                             const uintptr_t srcOffset){
    mHandle->asyncCopyFrom(source.mHandle, bytes, destOffset, srcOffset);
  }

  void memory::asyncCopyTo(void *dest,
                           const uintptr_t bytes,
                           const uintptr_t offset){
    mHandle->asyncCopyTo(dest, bytes, offset);
  }

  void memory::asyncCopyTo(memory &dest,
                           const uintptr_t bytes,
                           const uintptr_t destOffset,
                           const uintptr_t srcOffset){
    mHandle->asyncCopyTo(dest.mHandle, bytes, destOffset, srcOffset);
  }

  void memcpy(memory &dest,
              const void *source,
              const uintptr_t bytes,
              const uintptr_t offset){
    dest.copyFrom(source, bytes, offset);
  }

  void memcpy(memory &dest,
              const memory &source,
              const uintptr_t bytes,
              const uintptr_t destOffset,
              const uintptr_t srcOffset){
    dest.copyFrom(source, bytes, destOffset, srcOffset);
  }

  void memcpy(void *dest,
              memory &source,
              const uintptr_t bytes,
              const uintptr_t offset){
    source.copyTo(dest, bytes, offset);
  }

  void memcpy(memory &dest,
              memory &source,
              const uintptr_t bytes,
              const uintptr_t destOffset,
              const uintptr_t srcOffset){
    source.copyTo(dest, bytes, destOffset, srcOffset);
  }

  void asyncMemcpy(memory &dest,
                   const void *source,
                   const uintptr_t bytes,
                   const uintptr_t offset){
    dest.asyncCopyFrom(source, bytes, offset);
  }

  void asyncMemcpy(memory &dest,
                   const memory &source,
                   const uintptr_t bytes,
                   const uintptr_t destOffset,
                   const uintptr_t srcOffset){
    dest.asyncCopyFrom(source, bytes, destOffset, srcOffset);
  }

  void asyncMemcpy(void *dest,
                   memory &source,
                   const uintptr_t bytes,
                   const uintptr_t offset){
    source.asyncCopyTo(dest, bytes, offset);
  }

  void asyncMemcpy(memory &dest,
                   memory &source,
                   const uintptr_t bytes,
                   const uintptr_t destOffset,
                   const uintptr_t srcOffset){
    source.asyncCopyTo(dest, bytes, destOffset, srcOffset);
  }

  void memory::swap(memory &m){
    occa::mode mode2 = m.mode_;
    m.mode_        = mode_;
    mode_          = mode2;

    std::string strMode2 = m.strMode;
    m.strMode            = strMode;
    strMode              = strMode2;

    memory_v *mHandle2 = m.mHandle;
    m.mHandle          = mHandle;
    mHandle            = mHandle2;
  }

  void memory::free(){
    mHandle->free();
    delete mHandle;
  }
  //==================================


  //---[ Device ]---------------------
  device::device() :
    dHandle(NULL) {}

  device::device(const device &d) :
    mode_(d.mode_),
    strMode(d.strMode),
    dHandle(d.dHandle) {}

  device& device::operator = (const device &d){
    mode_   = d.mode_;
    dHandle = d.dHandle;

    return *this;
  }

  void device::setup(occa::mode m,
                     const int arg1, const int arg2){
    mode_   = m;
    strMode = modeToStr(m);

    switch(m){
    case Pthreads:
#if OCCA_PTHREADS_ENABLED
      dHandle = new device_t<Pthreads>(); break;
#else
      std::cout << "OCCA mode [Pthreads] is not enabled\n";
      throw 1;
#endif

    case OpenMP:
      dHandle = new device_t<OpenMP>(); break;

    case OpenCL:
#if OCCA_OPENCL_ENABLED
      dHandle = new device_t<OpenCL>(); break;
#else
      std::cout << "OCCA mode [OpenCL] is not enabled\n";
      throw 1;
#endif

    case CUDA:
#if OCCA_CUDA_ENABLED
      dHandle = new device_t<CUDA>(); break;
#else
      std::cout << "OCCA mode [CUDA] is not enabled\n";
      throw 1;
#endif

    case COI:
#if OCCA_COI_ENABLED
      dHandle = new device_t<COI>(); break;
#else
      std::cout << "OCCA mode [COI] is not enabled\n";
      throw 1;
#endif

    default:
      std::cout << "Incorrect OCCA mode given\n";
      throw 1;
    }

    dHandle->dev = this;
    dHandle->setup(arg1, arg2);

    currentStream = genStream();
  }

  void device::setup(const std::string &m,
                     const int arg1, const int arg2){
    setup(strToMode(m), arg1, arg2);
  }

  void device::setCompiler(const std::string &compiler_){
    dHandle->setCompiler(compiler_);
  }

  void device::setCompilerEnvScript(const std::string &compilerEnvScript_){
    dHandle->setCompilerEnvScript(compilerEnvScript_);
  }

  void device::setCompilerFlags(const std::string &compilerFlags_){
    dHandle->setCompilerFlags(compilerFlags_);
  }

  std::string& device::mode(){
    return strMode;
  }

  void device::flush(){
    dHandle->flush();
  }

  void device::finish(){
    dHandle->finish();
  }

  stream device::genStream(){
    streams.push_back( dHandle->genStream() );
    return streams.back();
  }

  stream device::getStream(){
    return currentStream;
  }

  void device::setStream(stream s){
    currentStream = s;
  }

  tag device::tagStream(){
    return dHandle->tagStream();
  }

  double device::timeBetween(const tag &startTag, const tag &endTag){
    return dHandle->timeBetween(startTag, endTag);
  }

  void device::free(stream s){
    dHandle->freeStream(s);
  }

  kernel device::buildKernelFromSource(const std::string &filename,
                                       const std::string &functionName,
                                       const kernelInfo &info_){
    kernel ker;

    ker.mode_   = mode_;
    ker.strMode = strMode;

    ker.kHandle      = dHandle->buildKernelFromSource(filename, functionName, info_);
    ker.kHandle->dev = this;

    return ker;
  }

  kernel device::buildKernelFromBinary(const std::string &filename,
                                       const std::string &functionName){
    kernel ker;

    ker.mode_   = mode_;
    ker.strMode = strMode;

    ker.kHandle      = dHandle->buildKernelFromBinary(filename, functionName);
    ker.kHandle->dev = this;

    return ker;
  }


  kernel device::buildKernelFromLoopy(const std::string &filename,
                                      const std::string &functionName,
                                      const int useLoopyOrFloopy){
    return buildKernelFromLoopy(filename, functionName, "", useLoopyOrFloopy);
  }

  kernel device::buildKernelFromLoopy(const std::string &filename,
                                      const std::string &functionName,
                                      const std::string &pythonCode,
                                      const int useLoopyOrFloopy){
    std::string cachedBinary = getCachedName(filename, pythonCode);

    struct stat buffer;
    bool fileExists = (stat(cachedBinary.c_str(), &buffer) == 0);

    if(fileExists){
      std::cout << "Found loo.py cached binary of [" << filename << "] in [" << cachedBinary << "]\n";
      return buildKernelFromBinary(cachedBinary, functionName);
    }

    std::string prefix, cacheName;

    getFilePrefixAndName(cachedBinary, prefix, cacheName);

    const std::string pCachedBinary = prefix + "p_" + cacheName;
    const std::string iCachedBinary = prefix + "i_" + cacheName;

    std::string loopyLang   = "loopy";
    std::string loopyHeader = pythonCode;

    if(useLoopyOrFloopy == occa::useFloopy){
      loopyHeader = "!$loopy begin transform\n" + loopyHeader + "\n!$loopy end transform\n";

      loopyLang = "floopy";
    }

    std::ofstream fs;
    fs.open(pCachedBinary.c_str());

    fs << loopyHeader << "\n\n" << readFile(filename);

    fs.close();

    std::stringstream command;

    command << "floopy --lang=" << loopyLang << " --target=cl:0,0 "
            << pCachedBinary << " " << iCachedBinary;

    const std::string &sCommand = command.str();

    std::cout << sCommand << '\n';

    system(sCommand.c_str());

    return buildKernelFromSource(iCachedBinary, functionName);
  }

  memory device::malloc(const uintptr_t bytes,
                        void *source){
    memory mem;

    mem.mode_   = mode_;
    mem.strMode = strMode;

    mem.mHandle      = dHandle->malloc(bytes, source);
    mem.mHandle->dev = this;

    return mem;
  }

  void device::free(){
    const int streamCount = streams.size();

    for(int i = 0; i < streamCount; ++i)
      dHandle->freeStream(streams[i]);

    dHandle->free();

    delete dHandle;
  }

  int device::simdWidth(){
    return dHandle->simdWidth();
  }
  //==================================
};
