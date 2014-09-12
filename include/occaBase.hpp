#ifndef OCCA_BASE_HEADER
#define OCCA_BASE_HEADER

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>

#include <xmmintrin.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <fcntl.h>

#include <sys/stat.h>
#include <sys/types.h>

#include "occaDefines.hpp"
#include "occaTools.hpp"

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
#  include <unistd.h>
#else
#  include <io.h>
#endif

#if OCCA_OPENCL_ENABLED
#  if   OCCA_OS == LINUX_OS
#    include <CL/cl.h>
#    include <CL/cl_gl.h>
#  elif OCCA_OS == OSX_OS
#    include <OpenCL/OpenCl.h>
#  endif
#endif

#if OCCA_CUDA_ENABLED
#  include <cuda.h>
#endif

namespace occa {
  class kernelInfo;
  extern kernelInfo defaultKernelInfo;

  //---[ Typedefs ]-------------------
  typedef void* stream;

  static const int CPU     = (1 << 0);
  static const int GPU     = (1 << 1);
  static const int FPGA    = (1 << 3);
  static const int XeonPhi = (1 << 2);
  static const int anyType = (CPU | GPU | FPGA | XeonPhi);

  static const int Intel     = (1 << 4);
  static const int AMD       = (1 << 5);
  static const int Altera    = (1 << 6);
  static const int NVIDIA    = (1 << 7);
  static const int anyVendor = (Intel | AMD | Altera | NVIDIA);

  const int any = (anyType | anyVendor);

  inline std::string deviceType(int type){
    if(type & CPU)     return "CPU";
    if(type & GPU)     return "GPU";
    if(type & FPGA)    return "FPGA";
    if(type & XeonPhi) return "Xeon Phi";

    return "N/A";
  }

  inline std::string vendor(int type){
    if(type & Intel)  return "Intel";
    if(type & AMD)    return "AMD";
    if(type & NVIDIA) return "NVIDIA";

    return "N/A";
  }

  static const uintptr_t useLoopy  = (1 << 0);
  static const uintptr_t useFloopy = (1 << 1);
  //==================================

  //---[ Mode ]-----------------------
  typedef int mode;

  static const occa::mode Pthreads = (1 << 20);
  static const occa::mode OpenMP   = (1 << 21);
  static const occa::mode OpenCL   = (1 << 22);
  static const occa::mode CUDA     = (1 << 23);
  static const occa::mode COI      = (1 << 24);


  static const occa::mode PthreadsIndex = 0;
  static const occa::mode OpenMPIndex   = 1;
  static const occa::mode OpenCLIndex   = 2;
  static const occa::mode CUDAIndex     = 3;
  static const occa::mode COIIndex      = 4;
  static const int modeCount = 5;

  inline std::string modeToStr(occa::mode m){
    if(m & Pthreads) return "Pthreads";
    if(m & OpenMP)   return "OpenMP";
    if(m & OpenCL)   return "OpenCL";
    if(m & CUDA)     return "CUDA";
    if(m & COI)      return "COI";

    OCCA_CHECK(false);

    return "No mode";
  }

  inline mode strToMode(const std::string &str){
    if(str.find("Pthreads") != std::string::npos) return Pthreads;
    if(str.find("OpenMP")   != std::string::npos) return OpenMP;
    if(str.find("OpenCL")   != std::string::npos) return OpenCL;
    if(str.find("CUDA")     != std::string::npos) return CUDA;
    if(str.find("COI")      != std::string::npos) return COI;

    OCCA_CHECK(false);

    return OpenMP;
  }

  inline std::string modes(int info, int preferredMode = 0){
    std::string ret = "";
    int info_ = info;
    int count = 0;

    if(preferredMode != 0){
      ret = "[" + modeToStr(preferredMode) + "]";
      info_ &= ~preferredMode;
      ++count;
    }

    if(info_ & Pthreads) ret += std::string(count++ ? ", " : "") + "Pthreads";
    if(info_ & OpenMP)   ret += std::string(count++ ? ", " : "") + "OpenMP";
    if(info_ & OpenCL)   ret += std::string(count++ ? ", " : "") + "OpenCL";
    if(info_ & CUDA)     ret += std::string(count++ ? ", " : "") + "CUDA";
    if(info_ & COI)      ret += std::string(count++ ? ", " : "") + "COI";

    if(count)
      return ret;
    else
      return "N/A";
  }
  //==================================


  class kernel_v;
  template <occa::mode> class kernel_t;
  class kernel;

  class memory_v;
  template <occa::mode> class memory_t;
  class memory;

  class device_v;
  template <occa::mode> class device_t;
  class device;

  //---[ Helper Classes ]-------------
  class deviceInfo {
  public:
    static const char *sLine, *dLine1, *dLine2;
    static const char *header;

    std::string name;
    int id, count, info;
    float memoryGB;
    int preferredMode;

    std::vector<std::string> labels, labelInfo;

    inline deviceInfo() :
      name("N/A"),
      id(0),
      count(1),
      info(0),
      memoryGB(0),
      preferredMode(0) {}

    inline deviceInfo(const deviceInfo &dInfo) :
      name(dInfo.name),
      id(dInfo.id),
      count(dInfo.count),
      info(dInfo.info),
      memoryGB(dInfo.memoryGB),
      preferredMode(dInfo.preferredMode),

      labels(dInfo.labels),
      labelInfo(dInfo.labelInfo) {}

    inline deviceInfo& operator = (const deviceInfo &dInfo){
      name          = dInfo.name;
      id            = dInfo.id;
      count         = dInfo.count;
      info          = dInfo.info;
      memoryGB      = dInfo.memoryGB;
      preferredMode = dInfo.preferredMode;

      labels    = dInfo.labels;
      labelInfo = dInfo.labelInfo;

      return *this;
    }

    inline std::string summarizedInfo() const {
      std::stringstream ss;

      ss << "| " << std::left << std::setw(42) << name
         << "| " << std::left << std::setw(4)  << count
         << "| " << std::left << std::setw(33) << modes(info, preferredMode)
         << "|";

      return ss.str();
    }

    inline std::string detailedInfo() const {
      std::stringstream ss;

      const int labelCount = labels.size();

      ss << dLine1 << '\n'
         << "| " << std::left << std::setw(55) << name << "|\n"
         << dLine1 << '\n'

         << "| " << std::left << std::setw(16) << "Device Count"
         << "| " << std::left << std::setw(37) << count << "|\n"
         << dLine2 << '\n'

         << "| " << std::left << std::setw(16) << "Vendor"
         << "| " << std::left << std::setw(37) << vendor(info) << "|\n"

         << "| " << std::left << std::setw(16) << "Memory"
         << "| " << std::left << std::setw(37) << memoryGB << "|\n"
         << dLine2 << '\n';

      for(int i = 0; i < labelCount; ++i)
        ss << "| " << std::left << std::setw(16) << labels[i]
           << "| " << std::left << std::setw(37) << labelInfo[i]
           << "|\n";

      ss << dLine1 << '\n';

      return ss.str();
    }

    inline bool operator == (const deviceInfo &dInfo) const {
      return ((name   == dInfo.name) &&
              (id     == dInfo.id)   &&
              (info   == dInfo.info));
    }

    inline bool operator != (const deviceInfo &info) const {
      return !(*this == info);
    }

    inline bool operator < (const deviceInfo &dInfo) const {
      if(name < dInfo.name) return true;
      if(name > dInfo.name) return false;

      if(id < dInfo.id) return true;
      if(id > dInfo.id) return false;

      if(info < dInfo.info) return true;

      return false;
    }

    inline bool operator > (const deviceInfo &info) const {
      return ((*this != info) && !(*this < info));
    }
  };

  class dim {
  public:
    union {
      struct {
        uintptr_t x, y, z;
      };
      uintptr_t data[3];
    };

    inline dim();
    inline dim(uintptr_t x_);
    inline dim(uintptr_t x_, uintptr_t y_);
    inline dim(uintptr_t x_, uintptr_t y_, uintptr_t z_);

    inline dim(const dim &d);

    inline dim& operator = (const dim &d);

    inline dim operator + (const dim &d);
    inline dim operator - (const dim &d);
    inline dim operator * (const dim &d);
    inline dim operator / (const dim &d);

    inline uintptr_t& operator [] (int i);
  };

  union kernelArg_t {
    int int_;
    unsigned int uint_;

    char char_;
    unsigned char uchar_;

    short short_;
    unsigned short ushort_;

    long long_;
    // unsigned long == uintptr_t

    float float_;
    double double_;

    uintptr_t uintptr_t_;
    void* void_;
  };

  class kernelArg {
  public:
    kernelArg_t arg;

    uintptr_t size;
    bool pointer;

    inline kernelArg(){
      arg.void_ = NULL;
    }

    inline kernelArg(kernelArg_t arg_, uintptr_t size_, bool pointer_) :
      size(size_),
      pointer(pointer_) {
      arg.void_ = arg_.void_;
    }

    inline kernelArg(const kernelArg &k) :
      size(k.size),
      pointer(k.pointer) {
      arg.void_ = k.arg.void_;
    }

    inline kernelArg& operator = (const kernelArg &k){
      arg.void_ = k.arg.void_;
      size      = k.size;
      pointer   = k.pointer;

      return *this;
    }

    OCCA_KERNEL_ARG_CONSTRUCTOR(int);
    OCCA_KERNEL_ARG_CONSTRUCTOR(char);
    OCCA_KERNEL_ARG_CONSTRUCTOR(short);
    OCCA_KERNEL_ARG_CONSTRUCTOR(long);

    OCCA_KERNEL_ARG_CONSTRUCTOR_ALIAS(unsigned int  , uint);
    OCCA_KERNEL_ARG_CONSTRUCTOR_ALIAS(unsigned char , uchar);
    OCCA_KERNEL_ARG_CONSTRUCTOR_ALIAS(unsigned short, ushort);

    OCCA_KERNEL_ARG_CONSTRUCTOR(float);
    OCCA_KERNEL_ARG_CONSTRUCTOR(double);

    // 32 bit: uintptr_t == unsigned int
#if OCCA_64_BIT
    OCCA_KERNEL_ARG_CONSTRUCTOR(uintptr_t);
#endif

    inline kernelArg(const occa::memory &m);

    inline kernelArg(void *arg_){
      arg.void_ = arg_;
      size = sizeof(void*);

      pointer = true;
    }

    inline void* data() const {
      return pointer ? arg.void_ : (void*) &arg;
    }
  };

  union tag {
    double tagTime;
#if OCCA_OPENCL_ENABLED
    cl_event clEvent;
#endif
#if OCCA_CUDA_ENABLED
    CUevent cuEvent;
#endif
  };
  //==================================


  //---[ Kernel ]---------------------
  class kernel_v {
    template <occa::mode> friend class occa::kernel_t;
    template <occa::mode> friend class occa::device_t;
    friend class occa::kernel;
    friend class occa::device;

  private:
    void* data;
    occa::device *dev;

    std::string functionName;

    int dims;
    dim inner, outer;

    int preferredDimSize_;

    void *startTime, *endTime;

  public:
    virtual inline ~kernel_v(){}

    virtual kernel_v* buildFromSource(const std::string &filename,
                                      const std::string &functionName_,
                                      const kernelInfo &info_ = defaultKernelInfo) = 0;

    virtual kernel_v* buildFromBinary(const std::string &filename,
                                      const std::string &functionName_) = 0;

    virtual int preferredDimSize() = 0;

#include "operators/occaVirtualOperatorDeclarations.hpp"

    virtual double timeTaken() = 0;

    virtual void free() = 0;
  };

  template <occa::mode mode>
  class kernel_t : public kernel_v {
  public:
    kernel_t();
    kernel_t(const std::string &filename,
             const std::string &functionName_,
             const kernelInfo &info_ = defaultKernelInfo);

    kernel_t(const kernel_t<mode> &k);
    kernel_t<mode>& operator = (const kernel_t<mode> &k);

    ~kernel_t();

    kernel_t<mode>* buildFromSource(const std::string &filename,
                                    const std::string &functionName_,
                                    const kernelInfo &info_ = defaultKernelInfo);

    kernel_t<mode>* buildFromBinary(const std::string &filename,
                                    const std::string &functionName_);

    int preferredDimSize();

#include "operators/occaOperatorDeclarations.hpp"

    double timeTaken();

    void free();
  };

  class kernel {
    friend class occa::device;

  private:
    occa::mode mode_;
    std::string strMode;

    kernel_v *kHandle;

    int argumentCount;
    kernelArg arguments[25];

  public:
    kernel();

    kernel(const kernel &k);
    kernel& operator = (const kernel &k);

    std::string& mode();

    kernel& buildFromSource(const std::string &filename,
                            const std::string &functionName_,
                            const kernelInfo &info_ = defaultKernelInfo);

    kernel& buildFromBinary(const std::string &filename,
                            const std::string &functionName_);

    int preferredDimSize();

    void setWorkingDims(int dims, dim inner, dim outer);

    void clearArgumentList();

    void addArgument(const int argPos,
                     const kernelArg &arg);

    void runFromArguments();

#include "operators/occaOperatorDeclarations.hpp"

    double timeTaken();

    void free();
  };
  //==================================


  //---[ Memory ]---------------------
  void memcpy(memory &dest,
              const void *source,
              const uintptr_t bytes = 0,
              const uintptr_t offset = 0);

  void memcpy(memory &dest,
              const memory &source,
              const uintptr_t bytes = 0,
              const uintptr_t destOffset = 0,
              const uintptr_t srcOffset = 0);

  void memcpy(void *dest,
              memory &source,
              const uintptr_t bytes = 0,
              const uintptr_t offset = 0);

  void memcpy(memory &dest,
              memory &source,
              const uintptr_t bytes = 0,
              const uintptr_t destOffset = 0,
              const uintptr_t srcOffset = 0);

  void asyncMemcpy(memory &dest,
                   const void *source,
                   const uintptr_t bytes = 0,
                   const uintptr_t offset = 0);

  void asyncMemcpy(memory &dest,
                   const memory &source,
                   const uintptr_t bytes = 0,
                   const uintptr_t destOffset = 0,
                   const uintptr_t srcOffset = 0);

  void asyncMemcpy(void *dest,
                   memory &source,
                   const uintptr_t bytes = 0,
                   const uintptr_t offset = 0);

  void asyncMemcpy(memory &dest,
                   memory &source,
                   const uintptr_t bytes = 0,
                   const uintptr_t destOffset = 0,
                   const uintptr_t srcOffset = 0);

  class memory_v {
    template <occa::mode> friend class occa::memory_t;
    template <occa::mode> friend class occa::device_t;
    friend class occa::memory;
    friend class occa::device;
    friend class occa::kernelArg;

  private:
    void *handle;
    occa::device *dev;

    uintptr_t size;

  public:
    virtual inline ~memory_v(){}

    virtual void copyFrom(const void *source,
                          const uintptr_t bytes = 0,
                          const uintptr_t offset = 0) = 0;

    virtual void copyFrom(const memory_v *source,
                          const uintptr_t bytes = 0,
                          const uintptr_t destOffset = 0,
                          const uintptr_t srcOffset = 0) = 0;

    virtual void copyTo(void *dest,
                        const uintptr_t bytes = 0,
                        const uintptr_t offset = 0) = 0;

    virtual void copyTo(memory_v *dest,
                        const uintptr_t bytes = 0,
                        const uintptr_t destOffset = 0,
                        const uintptr_t srcOffset = 0) = 0;

    virtual void asyncCopyFrom(const void *source,
                               const uintptr_t bytes = 0,
                               const uintptr_t offset = 0) = 0;

    virtual void asyncCopyFrom(const memory_v *source,
                               const uintptr_t bytes = 0,
                               const uintptr_t destOffset = 0,
                               const uintptr_t srcOffset = 0) = 0;

    virtual void asyncCopyTo(void *dest,
                             const uintptr_t bytes = 0,
                             const uintptr_t offset = 0) = 0;

    virtual void asyncCopyTo(memory_v *dest,
                             const uintptr_t bytes = 0,
                             const uintptr_t destOffset = 0,
                             const uintptr_t srcOffset = 0) = 0;

    virtual void free() = 0;
  };

  template <occa::mode mode>
  class memory_t : public memory_v {
    friend class occa::device_t<mode>;

  public:
    memory_t();

    memory_t(const memory_t &m);
    memory_t& operator = (const memory_t &m);

    inline ~memory_t(){};

    void copyFrom(const void *source,
                  const uintptr_t bytes = 0,
                  const uintptr_t offset = 0);

    void copyFrom(const memory_v *source,
                  const uintptr_t bytes = 0,
                  const uintptr_t destOffset = 0,
                  const uintptr_t srcOffset = 0);

    void copyTo(void *dest,
                const uintptr_t bytes = 0,
                const uintptr_t offset = 0);

    void copyTo(memory_v *dest,
                const uintptr_t bytes = 0,
                const uintptr_t destOffset = 0,
                const uintptr_t srcOffset = 0);

    void asyncCopyFrom(const void *source,
                       const uintptr_t bytes = 0,
                       const uintptr_t offset = 0);

    void asyncCopyFrom(const memory_v *source,
                       const uintptr_t bytes = 0,
                       const uintptr_t destOffset = 0,
                       const uintptr_t srcOffset = 0);

    void asyncCopyTo(void *dest,
                     const uintptr_t bytes = 0,
                     const uintptr_t offset = 0);

    void asyncCopyTo(memory_v *dest,
                     const uintptr_t bytes = 0,
                     const uintptr_t destOffset = 0,
                     const uintptr_t srcOffset = 0);

    void free();
  };


  class memory {
    friend class occa::device;
    friend class occa::kernelArg;

  private:
    occa::mode mode_;
    std::string strMode;

    memory_v *mHandle;

  public:
    memory();

    memory(const memory &m);
    memory& operator = (const memory &m);

    std::string& mode();

    inline uintptr_t bytes() const {
      if(mHandle == NULL)
        return 0;

      return mHandle->size;
    }

    void copyFrom(const void *source,
                  const uintptr_t bytes = 0,
                  const uintptr_t offset = 0);

    void copyFrom(const memory &source,
                  const uintptr_t bytes = 0,
                  const uintptr_t destOffset = 0,
                  const uintptr_t srcOffset = 0);

    void copyTo(void *dest,
                const uintptr_t bytes = 0,
                const uintptr_t offset = 0);

    void copyTo(memory &dest,
                const uintptr_t bytes = 0,
                const uintptr_t destOffset = 0,
                const uintptr_t srcOffset = 0);

    void asyncCopyFrom(const void *source,
                       const uintptr_t bytes = 0,
                       const uintptr_t offset = 0);

    void asyncCopyFrom(const memory &source,
                       const uintptr_t bytes = 0,
                       const uintptr_t destOffset = 0,
                       const uintptr_t srcOffset = 0);

    void asyncCopyTo(void *dest,
                     const uintptr_t bytes = 0,
                     const uintptr_t offset = 0);

    void asyncCopyTo(memory &dest,
                     const uintptr_t bytes = 0,
                     const uintptr_t destOffset = 0,
                     const uintptr_t srcOffset = 0);

    void swap(memory &m);

    void free();
  };

  //---[ KernelArg ]----------
  inline kernelArg::kernelArg(const occa::memory &m){
    arg.void_ = m.mHandle->handle;
    size = sizeof(void*);

    pointer = true;
  }
  //==================================


  //---[ Device ]---------------------
  template <occa::mode>
  std::vector<occa::deviceInfo> availableDevices();

  class device_v {
    template <occa::mode> friend class occa::device_t;
    template <occa::mode> friend class occa::kernel_t;

    friend class occa::device;
    friend class occa::memory;

  private:
    void* data;
    occa::device *dev;

    std::string compiler, compilerEnvScript, compilerFlags;

    int simdWidth_;

    uintptr_t memoryAllocated;

  public:
    virtual inline ~device_v(){}

    virtual void setup(const int arg1, const int arg2) = 0;

    virtual void getEnvironmentVariables() = 0;

    virtual void setCompiler(const std::string &compiler_) = 0;
    virtual void setCompilerEnvScript(const std::string &compilerEnvScript_) = 0;
    virtual void setCompilerFlags(const std::string &compilerFlags_) = 0;
    virtual std::string& getCompiler() = 0;
    virtual std::string& getCompilerEnvScript() = 0;
    virtual std::string& getCompilerFlags() = 0;

    virtual void flush()  = 0;
    virtual void finish() = 0;

    virtual stream genStream() = 0;
    virtual void freeStream(stream s) = 0;

    virtual tag tagStream() = 0;
    virtual double timeBetween(const tag &startTag, const tag &endTag) = 0;

    virtual kernel_v* buildKernelFromSource(const std::string &filename,
                                            const std::string &functionName_,
                                            const kernelInfo &info_ = defaultKernelInfo) = 0;

    virtual kernel_v* buildKernelFromBinary(const std::string &filename,
                                            const std::string &functionName_) = 0;

    virtual memory_v* malloc(const uintptr_t bytes,
                             void* source) = 0;

    virtual void free() = 0;

    virtual int simdWidth() = 0;
  };

  template <occa::mode mode>
  class device_t : public device_v {
    template <occa::mode> friend class occa::kernel_t;

  public:
    device_t();
    device_t(const int arg1, const int arg2);

    inline ~device_t(){}

    device_t(const device_t<mode> &k);
    device_t<mode>& operator = (const device_t<mode> &k);

    void setup(const int arg1, const int arg2);

    void getEnvironmentVariables();

    void setCompiler(const std::string &compiler_);
    void setCompilerEnvScript(const std::string &compilerEnvScript_);
    void setCompilerFlags(const std::string &compilerFlags_);
    std::string& getCompiler();
    std::string& getCompilerEnvScript();
    std::string& getCompilerFlags();

    void flush();
    void finish();

    stream genStream();
    void freeStream(stream s);

    tag tagStream();
    double timeBetween(const tag &startTag, const tag &endTag);

    kernel_v* buildKernelFromSource(const std::string &filename,
                                    const std::string &functionName,
									const kernelInfo &info_ = defaultKernelInfo);

    kernel_v* buildKernelFromBinary(const std::string &filename,
                                    const std::string &functionName);

    memory_v* malloc(const uintptr_t bytes,
                     void *source);

    void free();

    int simdWidth();
  };

  class device {
    template <occa::mode> friend class occa::kernel_t;
    template <occa::mode> friend class occa::memory_t;
    template <occa::mode> friend class occa::device_t;

    friend class occa::memory;

  private:
    occa::mode mode_;
    std::string strMode;

    device_v *dHandle;

    stream currentStream;
    std::vector<stream> streams;

  public:
    device();

    device(const device &d);
    device& operator = (const device &d);

    void setup(occa::mode m,
               const int arg1 = 0, const int arg2 = 0);
    void setup(const std::string &m,
               const int arg1 = 0, const int arg2 = 0);

    std::string& mode();

    void setCompiler(const std::string &compiler_);
    void setCompilerEnvScript(const std::string &compilerEnvScript_);
    void setCompilerFlags(const std::string &compilerFlags_);
    std::string& getCompiler();
    std::string& getCompilerEnvScript();
    std::string& getCompilerFlags();
    
    void flush();
    void finish();

    stream genStream();
    stream getStream();
    void setStream(stream s);

    tag tagStream();
    double timeBetween(const tag &startTag, const tag &endTag);

    uintptr_t memoryAllocated();

    void free(stream s);

    kernel buildKernelFromSource(const std::string &filename,
                                 const std::string &functionName,
                                 const kernelInfo &info_ = defaultKernelInfo);

    kernel buildKernelFromBinary(const std::string &filename,
                                 const std::string &functionName);

    kernel buildKernelFromLoopy(const std::string &filename,
                                const std::string &functionName,
                                int loopyOrFloopy);

    kernel buildKernelFromLoopy(const std::string &filename,
                                const std::string &functionName,
                                const std::string &pythonCode = "",
                                int loopyOrFloopy = occa::useLoopy);

    memory malloc(const uintptr_t bytes,
                  void *source = NULL);

    void free();

    int simdWidth();
  };
  //==================================

  class kernelInfo {
  public:
    std::string occaKeywords, header, flags;

    inline kernelInfo() :
      occaKeywords(""),
      header(""),
      flags("") {}

    inline kernelInfo(const kernelInfo &p) :
      occaKeywords(p.occaKeywords),
      header(p.header),
      flags(p.flags) {}

    inline kernelInfo& operator = (const kernelInfo &p){
      occaKeywords = p.occaKeywords;
      header = p.header;
      flags  = p.flags;

      return *this;
    }

    inline kernelInfo& operator += (const kernelInfo &p){
      header += p.header;
      flags  += p.flags;

      return *this;
    }

    inline std::string salt() const {
      return (header + flags);
    }

    inline static bool isAnOccaDefine(const std::string &name){
      if((name == "OCCA_USING_CPU") ||
         (name == "OCCA_USING_GPU") ||

         (name == "OCCA_USING_PTHREADS") ||
         (name == "OCCA_USING_OPENMP")   ||
         (name == "OCCA_USING_OPENCL")   ||
         (name == "OCCA_USING_CUDA")     ||
         (name == "OCCA_USING_COI")      ||

         (name == "occaInnerDim0") ||
         (name == "occaInnerDim1") ||
         (name == "occaInnerDim2") ||

         (name == "occaOuterDim0") ||
         (name == "occaOuterDim1") ||
         (name == "occaOuterDim2"))
        return true;

      return false;
    }

    inline void addOCCAKeywords(const std::string &keywords){
      occaKeywords = keywords;
    }

    inline void addIncludeDefine(const std::string &filename){
      header += "\n#include \"" + filename + "\"\n";
    }

    inline void addInclude(const std::string &filename){
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

      char *cFile = new char[fileSize + 1];
      cFile[fileSize] = '\0';

      ::read(fileHandle, cFile, fileSize);

      ::close(fileHandle);

      header += '\n';
      header += cFile;
      header += '\n';

      delete [] cFile;
    }

    template <class TM>
    inline void addDefine(const std::string &macro, const TM &value){
      std::stringstream ss;

      if(isAnOccaDefine(macro))
        ss << "#undef " << macro << "\n";

      ss << "#define " << macro << " " << value << '\n';

      header = ss.str() + header;
    }

    inline void addSource(const std::string &content){
      header += content;
    }

    inline void addCompilerFlag(const std::string &f){
      flags += " " + f;
    }

    inline void addCompilerIncludePath(const std::string &path){
#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
      flags += " -I \"" + path + "\"";
#else
      flags += " /I\"" + path + "\"";
#endif
    }
  };

  template <>
  inline void kernelInfo::addDefine(const std::string &macro, const std::string &value){
    std::stringstream ss;

    if(isAnOccaDefine(macro))
      ss << "#undef " << macro << "\n";

    // Make sure newlines are followed by escape characters
    std::string value2 = "";
    const int chars = value.size();

    for(int i = 0; i < chars; ++i){
      if(value[i] != '\n')
        value2 += value[i];
      else{
        if((i < (chars - 1))
           && (value[i] != '\\'))
          value2 += "\\\n";
        else
          value2 += '\n';
      }
    }

    if(value2[value2.size() - 1] != '\n')
      value2 += '\n';
    //======================================================

    ss << "#define " << macro << " " << value2 << '\n';

    header = ss.str() + header;
  }

  template <>
  inline void kernelInfo::addDefine(const std::string &macro, const float &value){
    std::stringstream ss;

    if(isAnOccaDefine(macro))
      ss << "#undef " << macro << "\n";

    ss << "#define " << macro << " ((float) " << std::setprecision(8) << value << ")\n";

    header = ss.str() + header;
  }

  template <>
  inline void kernelInfo::addDefine(const std::string &macro, const double &value){
    std::stringstream ss;

    if(isAnOccaDefine(macro))
      ss << "#undef " << macro << "\n";

    ss << "#define " << macro << " ((double) " << std::setprecision(16) << value << ")\n";

    header = ss.str() + header;
  }

  inline dim::dim() :
    x(1),
    y(1),
    z(1) {}

  inline dim::dim(uintptr_t x_) :
    x(x_),
    y(1),
    z(1) {}

  inline dim::dim(uintptr_t x_, uintptr_t y_) :
    x(x_),
    y(y_),
    z(1) {}

  inline dim::dim(uintptr_t x_, uintptr_t y_, uintptr_t z_) :
    x(x_),
    y(y_),
    z(z_) {}

  inline dim::dim(const dim &d) :
    x(d.x),
    y(d.y),
    z(d.z) {}

  inline dim& dim::operator = (const dim &d){
    x = d.x;
    y = d.y;
    z = d.z;

    return *this;
  }

  inline dim dim::operator + (const dim &d){
    return dim(x + d.x,
               y + d.y,
               z + d.z);
  }

  inline dim dim::operator - (const dim &d){
    return dim(x - d.x,
               y - d.y,
               z - d.z);
  }

  inline dim dim::operator * (const dim &d){
    return dim(x * d.x,
               y * d.y,
               z * d.z);
  }

  inline dim dim::operator / (const dim &d){
    return dim(x / d.x,
               y / d.y,
               z / d.z);
  }

  inline uintptr_t& dim::operator [] (int i){
    return data[i];
  }
};

#endif
