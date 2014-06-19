#ifndef OCCA_BASE_HEADER
#define OCCA_BASE_HEADER

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>

#include <xmmintrin.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>

#include <sys/stat.h>
#include <sys/types.h>

#include "occaDefines.hpp"
#include "occaTools.hpp"

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
  //==================================

  //---[ Mode ]-----------------------
  enum mode {Pthreads, OpenMP, OpenCL, CUDA, COI};

  inline std::string modeToStr(occa::mode m){
    switch(m){
    case Pthreads: return "Pthreads";
    case OpenMP  : return "OpenMP";
    case OpenCL  : return "OpenCL";
    case CUDA    : return "CUDA";
    case COI     : return "COI";
    }

    OCCA_CHECK(false);

    return "No mode";
  }

  inline mode strToMode(const std::string &str){
    if(str == "Pthreads") return Pthreads;
    if(str == "OpenMP")   return OpenMP;
    if(str == "OpenCL")   return OpenCL;
    if(str == "CUDA")     return CUDA;
    if(str == "COI")      return COI;

    OCCA_CHECK(false);

    return OpenMP;
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

  class dim {
  public:
    union {
      struct {
        size_t x, y, z;
      };
      size_t data[3];
    };

    inline dim();
    inline dim(size_t x_);
    inline dim(size_t x_, size_t y_);
    inline dim(size_t x_, size_t y_, size_t z_);

    inline dim(const dim &d);

    inline dim& operator = (const dim &d);
    inline dim  operator * (const dim &d);

    inline size_t& operator [] (int i);
  };

  union kernelArg_t {
    int int_;
    unsigned int uint_;

    char char_;
    unsigned char uchar_;

    short short_;
    unsigned short ushort_;

    long long_;
    // unsigned long == size_t

    float float_;
    double double_;

    size_t size_t_;
    void* void_;
  };

  class kernelArg {
  public:
    kernelArg_t arg;

    size_t size;
    bool pointer;

    inline kernelArg(){
      arg.void_ = NULL;
    }

    inline kernelArg(kernelArg_t arg_, size_t size_, bool pointer_) :
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

    OCCA_KERNEL_ARG_CONSTRUCTOR(size_t);

    inline kernelArg(occa::memory &m);

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

  //---[ Kernel ]---------------------
  class kernel_v {
    template<occa::mode> friend class occa::kernel_t;
    template<occa::mode> friend class occa::device_t;
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

    OCCA_VIRTUAL_KERNEL_OPERATOR_DECLARATIONS;

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

    OCCA_KERNEL_OPERATOR_DECLARATIONS;

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

    OCCA_KERNEL_OPERATOR_DECLARATIONS;

    double timeTaken();

    void free();
  };
  //==================================


  //---[ Memory ]---------------------
  void memcpy(memory &dest,
              const void *source,
              const size_t bytes = 0,
              const size_t offset = 0);

  void memcpy(memory &dest,
              const memory &source,
              const size_t bytes = 0,
              const size_t destOffset = 0,
              const size_t srcOffset = 0);

  void memcpy(void *dest,
              memory &source,
              const size_t bytes = 0,
              const size_t offset = 0);

  void memcpy(memory &dest,
              memory &source,
              const size_t bytes = 0,
              const size_t destOffset = 0,
              const size_t srcOffset = 0);

  void asyncMemcpy(memory &dest,
                   const void *source,
                   const size_t bytes = 0,
                   const size_t offset = 0);

  void asyncMemcpy(memory &dest,
                   const memory &source,
                   const size_t bytes = 0,
                   const size_t destOffset = 0,
                   const size_t srcOffset = 0);

  void asyncMemcpy(void *dest,
                   memory &source,
                   const size_t bytes = 0,
                   const size_t offset = 0);

  void asyncMemcpy(memory &dest,
                   memory &source,
                   const size_t bytes = 0,
                   const size_t destOffset = 0,
                   const size_t srcOffset = 0);

  class memory_v {
    template<occa::mode> friend class occa::memory_t;
    template<occa::mode> friend class occa::device_t;
    friend class occa::device;
    friend class occa::kernelArg;

  private:
    void *handle;
    occa::device *dev;

    size_t size;

  public:
    virtual inline ~memory_v(){}

    virtual void copyFrom(const void *source,
                          const size_t bytes = 0,
                          const size_t offset = 0) = 0;

    virtual void copyFrom(const memory_v *source,
                          const size_t bytes = 0,
                          const size_t destOffset = 0,
                          const size_t srcOffset = 0) = 0;

    virtual void copyTo(void *dest,
                        const size_t bytes = 0,
                        const size_t offset = 0) = 0;

    virtual void copyTo(memory_v *dest,
                        const size_t bytes = 0,
                        const size_t destOffset = 0,
                        const size_t srcOffset = 0) = 0;

    virtual void asyncCopyFrom(const void *source,
                               const size_t bytes = 0,
                               const size_t offset = 0) = 0;

    virtual void asyncCopyFrom(const memory_v *source,
                               const size_t bytes = 0,
                               const size_t destOffset = 0,
                               const size_t srcOffset = 0) = 0;

    virtual void asyncCopyTo(void *dest,
                             const size_t bytes = 0,
                             const size_t offset = 0) = 0;

    virtual void asyncCopyTo(memory_v *dest,
                             const size_t bytes = 0,
                             const size_t destOffset = 0,
                             const size_t srcOffset = 0) = 0;

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
                  const size_t bytes = 0,
                  const size_t offset = 0);

    void copyFrom(const memory_v *source,
                  const size_t bytes = 0,
                  const size_t destOffset = 0,
                  const size_t srcOffset = 0);

    void copyTo(void *dest,
                const size_t bytes = 0,
                const size_t offset = 0);

    void copyTo(memory_v *dest,
                const size_t bytes = 0,
                const size_t destOffset = 0,
                const size_t srcOffset = 0);

    void asyncCopyFrom(const void *source,
                       const size_t bytes = 0,
                       const size_t offset = 0);

    void asyncCopyFrom(const memory_v *source,
                       const size_t bytes = 0,
                       const size_t destOffset = 0,
                       const size_t srcOffset = 0);

    void asyncCopyTo(void *dest,
                     const size_t bytes = 0,
                     const size_t offset = 0);

    void asyncCopyTo(memory_v *dest,
                     const size_t bytes = 0,
                     const size_t destOffset = 0,
                     const size_t srcOffset = 0);

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

    void copyFrom(const void *source,
                  const size_t bytes = 0,
                  const size_t offset = 0);

    void copyFrom(const memory &source,
                  const size_t bytes = 0,
                  const size_t destOffset = 0,
                  const size_t srcOffset = 0);

    void copyTo(void *dest,
                const size_t bytes = 0,
                const size_t offset = 0);

    void copyTo(memory &dest,
                const size_t bytes = 0,
                const size_t destOffset = 0,
                const size_t srcOffset = 0);

    void asyncCopyFrom(const void *source,
                       const size_t bytes = 0,
                       const size_t offset = 0);

    void asyncCopyFrom(const memory &source,
                       const size_t bytes = 0,
                       const size_t destOffset = 0,
                       const size_t srcOffset = 0);

    void asyncCopyTo(void *dest,
                     const size_t bytes = 0,
                     const size_t offset = 0);

    void asyncCopyTo(memory &dest,
                     const size_t bytes = 0,
                     const size_t destOffset = 0,
                     const size_t srcOffset = 0);

    void swap(memory &m);

    void free();
  };

  //---[ KernelArg ]----------
  inline kernelArg::kernelArg(occa::memory &m){
    arg.void_ = m.mHandle->handle;
    size = sizeof(void*);

    pointer = true;
  }
  //==================================


  //---[ Device ]---------------------
  class device_v {
    template<occa::mode> friend class occa::device_t;
    template<occa::mode> friend class occa::kernel_t;
    friend class occa::device;

  private:
    void* data;
    occa::device *dev;

    std::string compiler, compilerFlags;

    int simdWidth_;

  public:
    virtual void setup(const int arg1, const int arg2) = 0;

    virtual void getEnvironmentVariables() = 0;

    virtual void setCompiler(const std::string &compiler) = 0;
    virtual void setCompilerFlags(const std::string &compilerFlags) = 0;

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

    virtual memory_v* malloc(const size_t bytes,
                             void* source) = 0;

    virtual void free() = 0;

    virtual int simdWidth() = 0;
  };

  template <occa::mode mode>
  class device_t : public device_v {
    template<occa::mode> friend class occa::kernel_t;

  private:
    size_t memoryUsed;

  public:
    device_t();
    device_t(const int arg1, const int arg2);

    device_t(const device_t<mode> &k);
    device_t<mode>& operator = (const device_t<mode> &k);

    void setup(const int arg1, const int arg2);

    void getEnvironmentVariables();

    void setCompiler(const std::string &compiler);
    void setCompilerFlags(const std::string &compilerFlags);

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

    memory_v* malloc(const size_t bytes,
                     void *source);

    void free();

    int simdWidth();
  };

  class device {
    template<occa::mode> friend class occa::kernel_t;
    template<occa::mode> friend class occa::memory_t;
    template<occa::mode> friend class occa::device_t;

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

    void setCompiler(const std::string &compiler);
    void setCompilerFlags(const std::string &compilerFlags);

    void flush();
    void finish();

    stream genStream();
    stream getStream();
    void setStream(stream s);

    tag tagStream();
    double timeBetween(const tag &startTag, const tag &endTag);

    kernel buildKernelFromSource(const std::string &filename,
                                 const std::string &functionName,
                                 const kernelInfo &info_ = defaultKernelInfo);

    kernel buildKernelFromBinary(const std::string &filename,
                                 const std::string &functionName);

    memory malloc(const size_t bytes,
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

      const size_t fileSize = fileInfo.st_size;

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

    ss << "#define " << macro << " " << std::setprecision(8) << value << '\n';

    header = ss.str() + header;
  }

  template <>
  inline void kernelInfo::addDefine(const std::string &macro, const double &value){
    std::stringstream ss;

    if(isAnOccaDefine(macro))
      ss << "#undef " << macro << "\n";

    ss << "#define " << macro << " " << std::setprecision(16) << value << '\n';

    header = ss.str() + header;
  }

  inline dim::dim() :
    x(1),
    y(1),
    z(1) {}

  inline dim::dim(size_t x_) :
    x(x_),
    y(1),
    z(1) {}

  inline dim::dim(size_t x_, size_t y_) :
    x(x_),
    y(y_),
    z(1) {}

  inline dim::dim(size_t x_, size_t y_, size_t z_) :
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

  inline dim dim::operator * (const dim &d){
    return dim(x * d.x,
               y * d.y,
               z * d.z);
  }

  inline size_t& dim::operator [] (int i){
    return data[i];
  }
};

#endif
