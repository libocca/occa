#ifndef OCCA_BASE_HEADER
#define OCCA_BASE_HEADER

#include <algorithm>
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

#include "occaParserTools.hpp"

#if (OCCA_OS == LINUX_OS) || (OCCA_OS == OSX_OS)
#  include <unistd.h>
#else
#  include "occaWinDefines.hpp"
#  include <io.h>
#endif

#if (OCCA_OPENCL_ENABLED)
#  if   (OCCA_OS == LINUX_OS)
#    include <CL/cl.h>
#    include <CL/cl_gl.h>
#  elif (OCCA_OS == OSX_OS)
#    include <OpenCL/OpenCl.h>
#  else
#    include "CL/opencl.h"
#  endif
#endif

#if OCCA_CUDA_ENABLED
#  include <cuda.h>
#endif

namespace occa {
  typedef int mode;

  class kernel_v;
  template <occa::mode> class kernel_t;
  class kernel;

  class memory_v;
  template <occa::mode> class memory_t;
  class memory;

  class device_v;
  template <occa::mode> class device_t;
  class device;

  class kernelInfo;

  //---[ Globals & Flags ]------------
  extern kernelInfo defaultKernelInfo;

  extern bool verboseCompilation_f;

  void setVerboseCompilation(const bool value);

  class ptrRange_t {
  public:
    char *start, *end;

    inline ptrRange_t() :
      start(NULL),
      end(NULL) {}

    inline ptrRange_t(void *ptr, const uintptr_t bytes = 0) :
      start((char*) ptr),
      end((char*) ptr + bytes) {}

    inline ptrRange_t(const ptrRange_t &r) :
      start(r.start),
      end(r.end) {}

    inline ptrRange_t& operator = (const ptrRange_t &r){
      start = r.start;
      end   = r.end;

      return *this;
    }

    inline bool operator == (const ptrRange_t &r){
      return ((start <= r.start) && (r.start < end));
    }

    inline friend int operator < (const ptrRange_t &a, const ptrRange_t &b){
      return (a.start < b.start);
    }
  };

  typedef std::map<ptrRange_t     , occa::memory_v*> ptrRangeMap_t;
  typedef std::map<occa::memory_v*, bool>            memoryPtrMap_t;

  extern ptrRangeMap_t  uvaMap;
  extern memoryPtrMap_t dirtyManagedMap;

  void free(void *ptr);
  //==================================

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

  static const int any = (anyType | anyVendor);

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

  static const int useLoopy  = (1 << 0);
  static const int useFloopy = (1 << 1);
  //==================================

  //---[ Mode ]-----------------------
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

    OCCA_CHECK(false, "Mode [" << m << "] is not valid");

    return "No mode";
  }

  inline mode strToMode(const std::string &str){
    const std::string upStr = upString(str);

    if(upStr == "PTHREADS") return Pthreads;
    if(upStr == "OPENMP")   return OpenMP;
    if(upStr == "OPENCL")   return OpenCL;
    if(upStr == "CUDA")     return CUDA;
    if(upStr == "COI")      return COI;

    OCCA_CHECK(false, "Mode [" << str << "] is not valid");

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

  class argInfo {
  public:
    std::string info, value;

    argInfo();

    argInfo(const argInfo &ai);
    argInfo& operator = (const argInfo &ai);

    argInfo(const std::string &info_);
    argInfo(const std::string &info_,
            const std::string &value_);

    template <class TM>
    argInfo operator = (const TM &value_) const {
      return argInfo(info, toString(value));
    }
  };

  static const argInfo platformID("platformID");
  static const argInfo deviceID("deviceID");

  static const argInfo schedule("schedule");
  static const argInfo chunk("chunk");

  static const argInfo threadCount("threadCount");
  static const argInfo pinnedCores("pinnedCores");

  class argInfoMap {
  public:
    std::map<std::string, std::string> iMap;

    inline argInfoMap(){}

    inline argInfoMap(const std::string &infos){
      parserNS::strNode *n;

      n = parserNS::splitContent(infos);
      n = parserNS::labelCode(n);

      while(n){
        std::string &info = n->value;
        std::string value;

        n = n->right;

        if((info != "mode")        &&
           (info != "platformID")  &&
           (info != "deviceID")    &&
           (info != "schedule")    &&
           (info != "chunk")       &&
           (info != "threadCount") &&
           (info != "schedule")    &&
           (info != "pinnedCores")){

          std::cout << "Flag [" << info << "] is not available, skipping it\n";

          while(n && (n->value != ","))
            n = n->right;

          if(n)
            n = n->right;

          continue;
        }

        if(n == NULL)
          break;

        if(n->value == "=")
          n = n->right;

        while(n && (n->value != ",")){
          std::string &v = n->value;

          occa::strip(v);

          if(v.size()){
            if(segmentPair(v[0]) == 0){
              value += v;
              value += ' ';
            }
            else if(n->down){
              std::string dv = n->down->toString();
              occa::strip(dv);

              value += dv;
              value += ' ';
            }
          }

          n = n->right;
        }

        if(n)
          n = n->right;

        occa::strip(value);

        iMap[info] = value;

        info  = "";
        value = "";
      }
    }

    inline bool has(const std::string &info){
      return (iMap.find(info) != iMap.end());
    }

    template <class TM>
    inline void set(const std::string &info, const TM &value){
      iMap[info] = toString(value);
    }

    inline std::string get(const std::string &info){
      std::map<std::string,std::string>::iterator it = iMap.find(info);

      if(it != iMap.end())
        return it->second;

      return "";
    }

    inline int iGet(const std::string &info){
      std::map<std::string,std::string>::iterator it = iMap.find(info);

      if(it != iMap.end())
        return atoi((it->second).c_str());

      return 0;
    }

    inline void iGets(const std::string &info, std::vector<int> &entries){
      std::map<std::string,std::string>::iterator it = iMap.find(info);

      if(it == iMap.end())
        return;

      const char *c = (it->second).c_str();

      while(*c != '\0'){
        skipWhitespace(c);

        if(isANumber(c)){
          entries.push_back(atoi(c));
          skipNumber(c);
        }
        else
          ++c;
      }
    }
  };

  template <>
  inline void argInfoMap::set(const std::string &info, const std::string &value){
    iMap[info] = value;
  }

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

    inline bool hasNegativeEntries();

    inline uintptr_t& operator [] (int i);
    inline uintptr_t operator [] (int i) const;
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
    occa::device *dev;

    kernelArg_t arg, arg2;

    uintptr_t size;
    bool pointer, hasTwoArgs;

    inline kernelArg(){
      dev        = NULL;
      arg.void_  = NULL;
      hasTwoArgs = false;
    }

    inline kernelArg(kernelArg_t arg_, uintptr_t size_, bool pointer_) :
      dev(NULL),
      size(size_),
      pointer(pointer_),
      hasTwoArgs(false) {
      arg.void_ = arg_.void_;
    }

    inline kernelArg(const kernelArg &k) :
      dev(k.dev),
      size(k.size),
      pointer(k.pointer),
      hasTwoArgs(k.hasTwoArgs) {
      arg.void_ = k.arg.void_;
    }

    inline kernelArg& operator = (const kernelArg &k){
      dev        = k.dev;
      arg.void_  = k.arg.void_;
      size       = k.size;
      pointer    = k.pointer;
      hasTwoArgs = k.hasTwoArgs;

      return *this;
    }

    template <class TM>
    inline kernelArg(const TM &arg_){
      dev = NULL;

      arg.void_ = const_cast<TM*>(&arg_);
      size = sizeof(TM);

      pointer    = true;
      hasTwoArgs = false;
    }

    template <class TM> inline kernelArg(TM *arg_);
    template <class TM> inline kernelArg(const TM *carg_);

    inline void* data() const {
      return pointer ? arg.void_ : (void*) &arg;
    }

    inline void markDirty() const {
      if(pointer){
        ptrRangeMap_t::iterator it = uvaMap.find(arg.void_);

        if(it != uvaMap.end())
          dirtyManagedMap[it->second] = true;
      }
    }
  };

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

  union tag {
    double tagTime;
#if OCCA_OPENCL_ENABLED
    cl_event clEvent;
#endif
#if OCCA_CUDA_ENABLED
    CUevent cuEvent;
#endif
  };

  struct textureInfo_t {
    void *arg;
    int dim;
    int bytesInEntry;
    uintptr_t w, h, d;
  };

  extern const int uint8FormatIndex;
  extern const int uint16FormatIndex;
  extern const int uint32FormatIndex;
  extern const int int8FormatIndex;
  extern const int int16FormatIndex;
  extern const int int32FormatIndex;
  extern const int halfFormatIndex;
  extern const int floatFormatIndex;

  extern const int sizeOfFormats[8];

  class formatType {
  private:
    int format_;
    int count_;

  public:
    formatType(const int format__, const int count__);

    formatType(const formatType &ft);
    formatType& operator = (const formatType &ft);

    template <occa::mode>
    void* format() const;

    int count() const;
    size_t bytes() const;
  };

  extern const int readOnly, readWrite;
  extern const occa::formatType uint8Format , uint8x2Format , uint8x4Format;
  extern const occa::formatType uint16Format, uint16x2Format, uint16x4Format;
  extern const occa::formatType uint32Format, uint32x2Format, uint32x4Format;
  extern const occa::formatType int8Format  , int8x2Format  , int8x4Format;
  extern const occa::formatType int16Format , int16x2Format , int16x4Format;
  extern const occa::formatType int32Format , int32x2Format , int32x4Format;
  extern const occa::formatType halfFormat  , halfx2Format  , halfx4Format;
  extern const occa::formatType floatFormat , floatx2Format , floatx4Format;
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

    int preferredDimSize_;

    void *startTime, *endTime;

    int dims;
    dim inner, outer;

    int nestedKernelCount;
    kernel *nestedKernels;

  public:
    virtual inline ~kernel_v(){}

    virtual std::string getCachedBinaryName(const std::string &filename,
                                            kernelInfo &info_) = 0;

    virtual kernel_v* buildFromSource(const std::string &filename,
                                      const std::string &functionName_,
                                      const kernelInfo &info_ = defaultKernelInfo) = 0;

    virtual kernel_v* buildFromBinary(const std::string &filename,
                                      const std::string &functionName_) = 0;

    virtual kernel_v* loadFromLibrary(const char *cache,
                                      const std::string &functionName_) = 0;

    virtual int preferredDimSize() = 0;

#include "operators/occaVirtualOperatorDeclarations.hpp"

    virtual double timeTaken() = 0;
    virtual double timeTakenBetween(void *start, void *end) = 0;

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

    std::string getCachedBinaryName(const std::string &filename,
                                    kernelInfo &info_);

    kernel_t<mode>* buildFromSource(const std::string &filename,
                                    const std::string &functionName_,
                                    const kernelInfo &info_ = defaultKernelInfo);

    kernel_t<mode>* buildFromBinary(const std::string &filename,
                                    const std::string &functionName_);

    kernel_t<mode>* loadFromLibrary(const char *cache,
                                    const std::string &functionName_);

    int preferredDimSize();

#include "operators/occaOperatorDeclarations.hpp"

    double timeTaken();
    double timeTakenBetween(void *start, void *end);

    void free();
  };

  class kernel {
    friend class occa::device;

  private:
    occa::mode mode_;
    std::string strMode;

    kernel_v *kHandle;

    int argumentCount;
    kernelArg arguments[OCCA_MAX_ARGS];

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

    kernel& loadFromLibrary(const char *cache,
                            const std::string &functionName_);

    int preferredDimSize();

    void setWorkingDims(int dims, dim inner, dim outer);

    void clearArgumentList();

    void addArgument(const int argPos,
                     const kernelArg &arg);

    void runFromArguments();

#include "operators/occaOperatorDeclarations.hpp"

    double timeTaken();
    double timeTakenBetween(void *start, void *end);

    void free();
  };

  class kernelDatabase {
  public:
    std::string kernelName;

    int modelKernelCount;
    std::vector<char> modelKernelAvailable;

    int kernelCount;
    std::vector<kernel> kernels;
    std::vector<char> kernelAllocated;

    kernelDatabase();
    kernelDatabase(const std::string kernelName_);

    kernelDatabase(const kernelDatabase &kdb);
    kernelDatabase& operator = (const kernelDatabase &kdb);

    void modelKernelIsAvailable(const int id);

    void addKernel(device d, kernel k);
    void addKernel(const int id, kernel k);

    void loadKernelFromLibrary(device &d);

    kernel& operator [] (device &d);

#include "operators/occaOperatorDeclarations.hpp"
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
    void *handle, *mappedPtr, *uvaPtr;
    occa::device *dev;

    uintptr_t size;

    bool isTexture;
    occa::textureInfo_t textureInfo;

    bool isMapped;
    bool isAWrapper;

  public:
    virtual inline ~memory_v(){}

    virtual void* getMemoryHandle() = 0;
    virtual void* getTextureHandle() = 0;

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

    virtual void mappedFree() = 0;

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

    void* getMemoryHandle();
    void* getTextureHandle();

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

    void mappedFree();

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

    void* textureArg() const;

    void* getMappedPointer();
    void* getMemoryHandle();
    void* getTextureHandle();

    void placeInUVA();
    void manage();

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
  template <class TM>
  inline kernelArg::kernelArg(TM *arg_){
    ptrRangeMap_t::iterator it = uvaMap.find(arg_);

    if(it == uvaMap.end()){
      dev = NULL;

      arg.void_ = arg_;
      size      = sizeof(TM*);

      pointer    = true;
      hasTwoArgs = false;
    }
    else{
      occa::memory_v *mem = it->second;

      dev = mem->dev;

      arg.void_ = mem;
      size      = sizeof(void*);

      pointer    = true;
      hasTwoArgs = false;
    }
  }

  template <class TM>
  inline kernelArg::kernelArg(const TM *carg_){
    TM *arg_ = const_cast<TM*>(carg_);

    ptrRangeMap_t::iterator it = uvaMap.find(arg_);

    if(it == uvaMap.end()){
      dev = NULL;

      arg.void_ = arg_;
      size      = sizeof(TM*);

      pointer    = true;
      hasTwoArgs = false;
    }
    else{
      occa::memory_v *mem = it->second;

      dev = mem->dev;

      arg.void_ = mem;
      size      = sizeof(void*);

      pointer    = true;
      hasTwoArgs = false;
    }
  }

  template <>
  inline kernelArg::kernelArg(const occa::memory &m){
    dev = m.mHandle->dev;

    arg.void_ = m.mHandle->handle;
    size      = sizeof(void*);

    pointer    = true;
    hasTwoArgs = m.mHandle->isTexture;

    if(hasTwoArgs)
      arg2.void_ = m.textureArg();
  }
  //==================================


  //---[ Device ]---------------------
  class deviceIdentifier {
  public:
    typedef std::map<std::string,std::string> flagMap_t;
    typedef flagMap_t::iterator               flagMapIterator;
    typedef flagMap_t::const_iterator         cFlagMapIterator;

    occa::mode mode_;
    flagMap_t flagMap;

    deviceIdentifier();

    deviceIdentifier(occa::mode m,
                     const char *c, const size_t chars);

    deviceIdentifier(occa::mode m, const std::string &s);

    deviceIdentifier(const deviceIdentifier &di);
    deviceIdentifier& operator = (const deviceIdentifier &di);

    void load(const char *c, const size_t chars);
    void load(const std::string &s);

    std::string flattenFlagMap() const;

    int compare(const deviceIdentifier &b) const;

    inline friend bool operator < (const deviceIdentifier &a,
                                   const deviceIdentifier &b){
      return (a.compare(b) < 0);
    }
  };

  template <occa::mode>
  std::vector<occa::deviceInfo> availableDevices();

#if OCCA_OPENCL_ENABLED
  namespace cl {
    occa::device wrapDevice(cl_platform_id platformID,
                            cl_device_id deviceID,
                            cl_context context);
  };
#endif

#if OCCA_CUDA_ENABLED
  namespace cuda {
    occa::device wrapDevice(CUdevice device, CUcontext context);
  };
#endif

#if OCCA_COI_ENABLED
  namespace coi {
    occa::device wrapDevice(COIENGINE coiDevice);
  };
#endif

  class device_v {
    template <occa::mode> friend class occa::device_t;
    template <occa::mode> friend class occa::kernel_t;
    friend class occa::device;

  private:
    void* data;
    occa::device *dev;

    std::string compiler, compilerEnvScript, compilerFlags;

    uintptr_t memoryAllocated;

    int simdWidth_;

  public:
    virtual inline ~device_v(){}

    virtual void setup(argInfoMap &aim) = 0;

    virtual void addOccaHeadersToInfo(kernelInfo &info) = 0;
    virtual std::string getInfoSalt(const kernelInfo &info) = 0;
    virtual deviceIdentifier getIdentifier() const = 0;

    virtual void getEnvironmentVariables() = 0;

    virtual void setCompiler(const std::string &compiler_) = 0;
    virtual void setCompilerEnvScript(const std::string &compilerEnvScript_) = 0;
    virtual void setCompilerFlags(const std::string &compilerFlags_) = 0;

    virtual std::string& getCompiler() = 0;
    virtual std::string& getCompilerEnvScript() = 0;
    virtual std::string& getCompilerFlags() = 0;

    virtual void flush()  = 0;
    virtual void finish() = 0;

    virtual void waitFor(tag tag_) = 0;

    virtual stream createStream() = 0;
    virtual void freeStream(stream s) = 0;
    virtual stream wrapStream(void *handle_) = 0;

    virtual tag tagStream() = 0;
    virtual double timeBetween(const tag &startTag, const tag &endTag) = 0;

    virtual kernel_v* buildKernelFromSource(const std::string &filename,
                                            const std::string &functionName_,
                                            const kernelInfo &info_ = defaultKernelInfo) = 0;

    virtual kernel_v* buildKernelFromBinary(const std::string &filename,
                                            const std::string &functionName_) = 0;

    virtual void cacheKernelInLibrary(const std::string &filename,
                                      const std::string &functionName_,
                                      const kernelInfo &info_ = defaultKernelInfo) = 0;

    virtual kernel_v* loadKernelFromLibrary(const char *cache,
                                            const std::string &functionName_) = 0;

#if OCCA_OPENCL_ENABLED
    friend occa::device cl::wrapDevice(cl_platform_id platformID,
                                       cl_device_id deviceID,
                                       cl_context context);
#endif

#if OCCA_CUDA_ENABLED
    friend occa::device cuda::wrapDevice(CUdevice device, CUcontext context);
#endif

#if OCCA_COI_ENABLED
    friend occa::device coi::wrapDevice(COIENGINE coiDevice);
#endif

    virtual memory_v* wrapMemory(void *handle_,
                                 const uintptr_t bytes) = 0;

    virtual memory_v* wrapTexture(void *handle_,
                                  const int dim, const occa::dim &dims,
                                  occa::formatType type, const int permissions) = 0;

    virtual memory_v* malloc(const uintptr_t bytes,
                             void* source) = 0;

    virtual memory_v* textureAlloc(const int dim, const occa::dim &dims,
                                   void *source,
                                   occa::formatType type, const int permissions) = 0;

    virtual memory_v* mappedAlloc(const uintptr_t bytes,
                                  void *source) = 0;

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

    void setup(argInfoMap &aim);

    void addOccaHeadersToInfo(kernelInfo &info);
    std::string getInfoSalt(const kernelInfo &info);
    deviceIdentifier getIdentifier() const;

    void getEnvironmentVariables();

    static void appendAvailableDevices(std::vector<device> &dList);

    void setCompiler(const std::string &compiler_);
    void setCompilerEnvScript(const std::string &compilerEnvScript_);
    void setCompilerFlags(const std::string &compilerFlags_);

    std::string& getCompiler();
    std::string& getCompilerEnvScript();
    std::string& getCompilerFlags();

    void flush();
    void finish();

    void waitFor(tag tag_);

    stream createStream();
    void freeStream(stream s);
    stream wrapStream(void *handle_);

    tag tagStream();
    double timeBetween(const tag &startTag, const tag &endTag);

    kernel_v* buildKernelFromSource(const std::string &filename,
                                    const std::string &functionName,
                                    const kernelInfo &info_ = defaultKernelInfo);

    kernel_v* buildKernelFromBinary(const std::string &filename,
                                    const std::string &functionName);

    void cacheKernelInLibrary(const std::string &filename,
                              const std::string &functionName_,
                              const kernelInfo &info_ = defaultKernelInfo);

    kernel_v* loadKernelFromLibrary(const char *cache,
                                    const std::string &functionName_);

#if OCCA_OPENCL_ENABLED
    friend occa::device cl::wrapDevice(cl_platform_id platformID,
                                       cl_device_id deviceID,
                                       cl_context context);
#endif

#if OCCA_CUDA_ENABLED
    friend occa::device cuda::wrapDevice(CUdevice device, CUcontext context);
#endif

#if OCCA_COI_ENABLED
    friend occa::device coi::wrapDevice(COIENGINE coiDevice);
#endif

    memory_v* wrapMemory(void *handle_,
                         const uintptr_t bytes);

    memory_v* wrapTexture(void *handle_,
                          const int dim, const occa::dim &dims,
                          occa::formatType type, const int permissions);

    memory_v* malloc(const uintptr_t bytes,
                     void *source);

    memory_v* textureAlloc(const int dim, const occa::dim &dims,
                           void *source,
                           occa::formatType type, const int permissions);

    memory_v* mappedAlloc(const uintptr_t bytes,
                          void *source);

    void free();

    int simdWidth();
  };

  class device {
    template <occa::mode> friend class occa::kernel_t;
    template <occa::mode> friend class occa::memory_t;
    template <occa::mode> friend class occa::device_t;

    friend class occa::memory;
    friend class occa::kernelDatabase;

  private:
    occa::mode mode_;
    std::string strMode;

    int modelID_, id_;
    device_v *dHandle;

    stream currentStream;
    std::vector<stream> streams;

    uintptr_t bytesAllocated_;

  public:
    device();

    device(const device &d);
    device& operator = (const device &d);

    void setupHandle(occa::mode m);
    void setupHandle(const std::string &m);

    void setup(const std::string &infos);

    void setup(occa::mode m,
               const int arg1, const int arg2);
    void setup(occa::mode m,
               const argInfo &arg1);
    void setup(occa::mode m,
               const argInfo &arg1, const argInfo &arg2);
    void setup(occa::mode m,
               const argInfo &arg1, const argInfo &arg2, const argInfo &arg3);

    void setup(const std::string &m,
               const int arg1, const int arg2);
    void setup(const std::string &m,
               const argInfo &arg1);
    void setup(const std::string &m,
               const argInfo &arg1, const argInfo &arg2);
    void setup(const std::string &m,
               const argInfo &arg1, const argInfo &arg2, const argInfo &arg3);

    uintptr_t bytesAllocated() const;

    deviceIdentifier getIdentifier() const;

    int modelID();
    int id();

    int modeID();
    std::string& mode();

    void setCompiler(const std::string &compiler_);
    void setCompilerEnvScript(const std::string &compilerEnvScript_);
    void setCompilerFlags(const std::string &compilerFlags_);

    std::string& getCompiler();
    std::string& getCompilerEnvScript();
    std::string& getCompilerFlags();

    void flush();
    void finish();

    void waitFor(tag tag_);

    stream createStream();
    stream getStream();
    void setStream(stream s);
    stream wrapStream(void *handle_);

    tag tagStream();
    double timeBetween(const tag &startTag, const tag &endTag);

    void free(stream s);

    kernel buildKernelFromString(const std::string &content,
                                 const std::string &functionName,
                                 const bool useParser);

    kernel buildKernelFromString(const std::string &content,
                                 const std::string &functionName,
                                 const kernelInfo &info_ = defaultKernelInfo,
                                 const bool useParser    = true);

    kernel buildKernelFromSource(const std::string &filename,
                                 const std::string &functionName,
                                 const kernelInfo &info_ = defaultKernelInfo);

    kernel buildKernelFromBinary(const std::string &filename,
                                 const std::string &functionName);

    void cacheKernelInLibrary(const std::string &filename,
                              const std::string &functionName_,
                              const kernelInfo &info_ = defaultKernelInfo);

    kernel loadKernelFromLibrary(const char *cache,
                                 const std::string &functionName_);

    kernel buildKernelFromLoopy(const std::string &filename,
                                const std::string &functionName,
                                const int loopyOrFloopy);

    kernel buildKernelFromLoopy(const std::string &filename,
                                const std::string &functionName,
                                const kernelInfo &info_ = defaultKernelInfo,
                                const int loopyOrFloopy = occa::useLoopy);

    kernel& operator [] (kernelDatabase &kdb);

#if OCCA_OPENCL_ENABLED
    friend occa::device cl::wrapDevice(cl_platform_id platformID,
                                       cl_device_id deviceID,
                                       cl_context context);
#endif

#if OCCA_CUDA_ENABLED
    friend occa::device cuda::wrapDevice(CUdevice device, CUcontext context);
#endif

#if OCCA_COI_ENABLED
    friend occa::device coi::wrapDevice(COIENGINE coiDevice);
#endif

    memory wrapMemory(void *handle_,
                      const uintptr_t bytes);

    memory wrapTexture(void *handle_,
                       const int dim, const occa::dim &dims,
                       occa::formatType type, const int permissions);

    memory malloc(const uintptr_t bytes,
                  void *source = NULL);

    memory managedAlloc(const uintptr_t bytes,
                        void *source = NULL);

    void* uvaAlloc(const uintptr_t bytes,
                   void *source = NULL);

    void* managedUvaAlloc(const uintptr_t bytes,
                          void *source = NULL);

    memory textureAlloc(const int dim, const occa::dim &dims,
                        void *source,
                        occa::formatType type, const int permissions = readWrite);

    memory managedTextureAlloc(const int dim, const occa::dim &dims,
                               void *source,
                               occa::formatType type, const int permissions = readWrite);

    memory mappedAlloc(const uintptr_t bytes,
                       void *source = NULL);

    memory managedMappedAlloc(const uintptr_t bytes,
                              void *source = NULL);

    void free();

    int simdWidth();
  };

  extern mutex_t deviceListMutex;
  extern std::vector<device> deviceList;

  std::vector<device>& getDeviceList();
  //==================================


  //---[ Kernel Database ]------------
  inline kernel& kernelDatabase::operator [] (device &d){
    OCCA_CHECK(0 <= d.modelID_                 , "Device [modelID] is not set");
    OCCA_CHECK((d.modelID_ < modelKernelCount) , "Kernel is not compiled for chosen device");
    OCCA_CHECK(modelKernelAvailable[d.modelID_], "Kernel is not compiled for chosen device");
    OCCA_CHECK(0 <= d.id_                      , "Device not set");

    if((d.id_ < kernelCount) && kernelAllocated[d.id_])
      return kernels[d.id_];

    loadKernelFromLibrary(d);

    return kernels[d.id_];
  }

  inline kernel& device::operator [] (kernelDatabase &kdb){
    return kdb[*this];
  }
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
      header += '\n';
      header += readFile(filename);
      header += '\n';
    }

    inline void removeDefine(const std::string &macro){
      if(!isAnOccaDefine(macro))
        header += "#undef " + macro + '\n';
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
      flags += " /I \"" + path + "\"";
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

  inline bool dim::hasNegativeEntries(){
    return ((x & (1 << (sizeof(uintptr_t) - 1))) ||
            (y & (1 << (sizeof(uintptr_t) - 1))) ||
            (z & (1 << (sizeof(uintptr_t) - 1))));
  }

  inline uintptr_t& dim::operator [] (int i){
    return data[i];
  }

  inline uintptr_t dim::operator [] (int i) const {
    return data[i];
  }
};

#endif
