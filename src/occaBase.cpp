#include "occa.hpp"

// Use events for timing!

namespace occa {
  kernelInfo defaultKernelInfo;

  //---[ Kernel ]---------------------
  kernel::kernel() :
    mode_(),
    kHandle(NULL) {}

  kernel::kernel(const kernel &k) :
    mode_(k.mode_),
    kHandle(k.kHandle) {}

  kernel& kernel::operator = (const kernel &k){
    mode_   = k.mode_;
    kHandle = k.kHandle;

    return *this;
  }

  std::string kernel::mode(){
    return modeToStr(mode_);
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
    switch(argumentCount){
      OCCA_RUN_FROM_ARGUMENTS_SWITCH;
    }

    return;
  }

  OCCA_KERNEL_OPERATOR_DEFINITIONS;

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
    mHandle(NULL) {}

  memory::memory(const memory &m) :
    mode_(m.mode_),
    mHandle(m.mHandle) {}

  memory& memory::operator = (const memory &m){
    mode_   = m.mode_;
    mHandle = m.mHandle;

    return *this;
  }

  std::string memory::mode(){
    return modeToStr(mode_);
  }

  void memory::copyFrom(const void *source,
                        const size_t bytes,
                        const size_t offset){
    mHandle->copyFrom(source, bytes, offset);
  }

  void memory::copyFrom(const memory &source,
                        const size_t bytes,
                        const size_t offset){
    mHandle->copyFrom(source.mHandle, bytes, offset);
  }

  void memory::copyTo(void *dest,
                      const size_t bytes,
                      const size_t offset){
    mHandle->copyTo(dest, bytes, offset);
  }

  void memory::copyTo(memory &dest,
                      const size_t bytes,
                      const size_t offset){
    mHandle->copyTo(dest.mHandle, bytes, offset);
  }

  void memory::asyncCopyFrom(const void *source,
                             const size_t bytes,
                             const size_t offset){
    mHandle->asyncCopyFrom(source, bytes, offset);
  }

  void memory::asyncCopyFrom(const memory &source,
                             const size_t bytes,
                             const size_t offset){
    mHandle->asyncCopyFrom(source.mHandle, bytes, offset);
  }

  void memory::asyncCopyTo(void *dest,
                           const size_t bytes,
                           const size_t offset){
    mHandle->asyncCopyTo(dest, bytes, offset);
  }

  void memory::asyncCopyTo(memory &dest,
                           const size_t bytes,
                           const size_t offset){
    mHandle->asyncCopyTo(dest.mHandle, bytes, offset);
  }

  void memory::swap(memory &m){
    occa::mode mode2 = m.mode_;
    m.mode_        = mode_;
    mode_          = mode2;

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
    dHandle(d.dHandle) {}

  device& device::operator = (const device &d){
    mode_   = d.mode_;
    dHandle = d.dHandle;

    return *this;
  }

  void device::setup(occa::mode m,
                     const int arg1, const int arg2){
    mode_ = m;

    switch(m){
    case Pthreads:
      dHandle = new device_t<Pthreads>(); break;

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

  void device::setCompiler(const std::string &compiler){
    dHandle->setCompiler(compiler);
  }

  void device::setCompilerFlags(const std::string &compilerFlags){
    dHandle->setCompilerFlags(compilerFlags);
  }

  std::string device::mode(){
    return modeToStr(mode_);
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

  kernel device::buildKernelFromSource(const std::string &filename,
                                       const std::string &functionName,
                                       const kernelInfo &info_){
    kernel ker;

    ker.mode_ = mode_;

    ker.kHandle      = dHandle->buildKernelFromSource(filename, functionName, info_);
    ker.kHandle->dev = this;

    return ker;
  }

  kernel device::buildKernelFromBinary(const std::string &filename,
                                       const std::string &functionName){
    kernel ker;

    ker.mode_ = mode_;

    ker.kHandle      = dHandle->buildKernelFromBinary(filename, functionName);
    ker.kHandle->dev = this;

    return ker;
  }

  memory device::malloc(const size_t bytes,
                        void *source){
    memory mem;

    mem.mode_ = mode_;

    mem.mHandle      = dHandle->malloc(bytes, source);
    mem.mHandle->dev = this;

    return mem;
  }

  void device::free(){
    const int streamCount = streams.size();

    for(int i = 0; i < streamCount; ++i)
      dHandle->freeStream(streams[i]);
  }

  int device::simdWidth(){
    return dHandle->simdWidth();
  }
  //==================================
};
