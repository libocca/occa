#ifndef OCCA_BASE_HEADER
#define OCCA_BASE_HEADER

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <xmmintrin.h>

#include <sys/stat.h>
#include <sys/types.h>

#include "occaDefines.hpp"
#include "occaTools.hpp"

namespace occa {
  extern std::string ompCompiler, ompCompilerFlags;
  extern std::string cudaCompiler, cudaArch, cudaCompilerFlags;

  class kernelInfo;
  extern kernelInfo defaultKernelInfo;

  //---[ Typedefs ]-------------------
  typedef void* stream;
  typedef void* tag;
  //==================================

  //---[ Mode ]-----------------------
  enum mode {OpenMP, OpenCL, CUDA};

  inline std::string modeToStr(occa::mode m){
    switch(m){
    case OpenMP: return "OpenMP";
    case OpenCL: return "OpenCL";
    case CUDA  : return "CUDA";
    }

    OCCA_CHECK(false);
    return "No mode";
  }

  inline mode strToMode(const std::string &str){
    if(str == "OpenMP") return OpenMP;
    if(str == "OpenCL") return OpenCL;
    if(str == "CUDA")   return CUDA;

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

  class kernelArg;

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
    kernel_v *kHandle;

  public:
    kernel();

    kernel(const kernel &k);
    kernel& operator = (const kernel &k);

    std::string mode();

    kernel& buildFromSource(const std::string &filename,
                            const std::string &functionName_,
                            const kernelInfo &info_ = defaultKernelInfo);

    kernel& buildFromBinary(const std::string &filename,
                            const std::string &functionName_);

    int preferredDimSize();

    void setWorkingDims(int dims, dim inner, dim outer);

    OCCA_KERNEL_OPERATOR_DECLARATIONS;

    double timeTaken();

    void free();
  };
  //==================================


  //---[ Memory ]---------------------
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
                          const size_t offset = 0) = 0;

    virtual void copyTo(void *dest,
                        const size_t bytes = 0,
                        const size_t offset = 0) = 0;

    virtual void copyTo(memory_v *dest,
                        const size_t bytes = 0,
                        const size_t offset = 0) = 0;

    virtual void asyncCopyFrom(const void *source,
                               const size_t bytes = 0,
                               const size_t offset = 0) = 0;

    virtual void asyncCopyFrom(const memory_v *source,
                               const size_t bytes = 0,
                               const size_t offset = 0) = 0;

    virtual void asyncCopyTo(void *dest,
                             const size_t bytes = 0,
                             const size_t offset = 0) = 0;

    virtual void asyncCopyTo(memory_v *dest,
                             const size_t bytes = 0,
                             const size_t offset = 0) = 0;

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
                  const size_t offset = 0);

    void copyTo(void *dest,
                const size_t bytes = 0,
                const size_t offset = 0);

    void copyTo(memory_v *dest,
                const size_t bytes = 0,
                const size_t offset = 0);

    void asyncCopyFrom(const void *source,
                       const size_t bytes = 0,
                       const size_t offset = 0);

    void asyncCopyFrom(const memory_v *source,
                       const size_t bytes = 0,
                       const size_t offset = 0);

    void asyncCopyTo(void *dest,
                     const size_t bytes = 0,
                     const size_t offset = 0);

    void asyncCopyTo(memory_v *dest,
                     const size_t bytes = 0,
                     const size_t offset = 0);

    void free();
  };


  class memory {
    friend class occa::device;
    friend class occa::kernelArg;

  private:
    occa::mode mode_;
    memory_v *mHandle;

  public:
    memory();

    memory(const memory &m);
    memory& operator = (const memory &m);

    std::string mode();

    void copyFrom(const void *source,
                  const size_t bytes = 0,
                  const size_t offset = 0);

    void copyFrom(const memory &source,
                  const size_t bytes = 0,
                  const size_t offset = 0);

    void copyTo(void *dest,
                const size_t bytes = 0,
                const size_t offset = 0);

    void copyTo(memory &dest,
                const size_t bytes = 0,
                const size_t offset = 0);

    void asyncCopyFrom(const void *source,
                       const size_t bytes = 0,
                       const size_t offset = 0);

    void asyncCopyFrom(const memory &source,
                       const size_t bytes = 0,
                       const size_t offset = 0);

    void asyncCopyTo(void *dest,
                     const size_t bytes = 0,
                     const size_t offset = 0);

    void asyncCopyTo(memory &dest,
                     const size_t bytes = 0,
                     const size_t offset = 0);

    void swap(memory &m);

    void free();
  };
  //==================================


  //---[ Device ]---------------------
  class device_v {
    template<occa::mode> friend class occa::device_t;
    friend class occa::device;

  private:
    void* data;
    occa::device *dev;

    int simdWidth_;

  public:
    virtual void setup(const int platform, const int device) = 0;

    virtual void flush()  = 0;
    virtual void finish() = 0;

    virtual stream genStream() = 0;
    virtual void freeStream(stream s) = 0;

    virtual kernel_v* buildKernelFromSource(const std::string &filename,
                                            const std::string &functionName_,
                                            const kernelInfo &info_ = defaultKernelInfo) = 0;

    virtual kernel_v* buildKernelFromBinary(const std::string &filename,
                                            const std::string &functionName_) = 0;

    virtual memory_v* malloc(const size_t bytes,
                             void* source) = 0;

    virtual int simdWidth() = 0;
  };

  template <occa::mode mode>
  class device_t : public device_v {
  private:
    void *data;
    size_t memoryUsed;

  public:
    device_t();
    device_t(const int platform, const int device);

    device_t(const device_t<mode> &k);
    device_t<mode>& operator = (const device_t<mode> &k);

    void setup(const int platform, const int device);

    void flush();
    void finish();

    stream genStream();
    void freeStream(stream s);

    kernel_v* buildKernelFromSource(const std::string &filename,
                                    const std::string &functionName,
                                    const kernelInfo &info_ = defaultKernelInfo);

    kernel_v* buildKernelFromBinary(const std::string &filename,
                                    const std::string &functionName);

    memory_v* malloc(const size_t bytes,
                     void *source);

    int simdWidth();
  };

  class device {
    template<occa::mode> friend class occa::kernel_t;
    template<occa::mode> friend class occa::memory_t;
    template<occa::mode> friend class occa::device_t;

  private:
    occa::mode mode_;
    device_v *dHandle;

    stream currentStream;
    std::vector<stream> streams;

  public:
    std::string ompCompiler, ompCompilerFlags;
    std::string cudaCompiler, cudaArch, cudaCompilerFlags;

    device();

    device(const device &d);
    device& operator = (const device &d);

    void setup(occa::mode m,
               const int platform, const int device);
    void setup(const std::string &m,
               const int platform, const int device);

    std::string mode();

    void flush();
    void finish();

    stream genStream();
    stream getStream();
    void setStream(stream s);

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

    inline void addOCCAKeywords(const std::string &keywords){
      occaKeywords = keywords;
    }

    inline void addInclude(const std::string &file){
      header += ("#include \"" + file + "\"\n");
    }

    template <class TM>
    inline void addDefine(const std::string &macro, const TM &value){
      std::stringstream ss;
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
