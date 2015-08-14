#ifndef OCCA_BASE_HEADER
#define OCCA_BASE_HEADER

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>

#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <fcntl.h>

#include <sys/stat.h>
#include <sys/types.h>

#if OCCA_SSE
#  include <xmmintrin.h>
#endif

#include "occa/defines.hpp"
#include "occa/tools.hpp"
#include "occa/uva.hpp"

#include "occa/parser/tools.hpp"

#if (OCCA_OS & (LINUX_OS | OSX_OS))
#  include <unistd.h>
#else
#  include <io.h>
#endif

#if (OCCA_OPENCL_ENABLED)
#  if   (OCCA_OS & LINUX_OS)
#    include <CL/cl.h>
#    include <CL/cl_gl.h>
#  elif (OCCA_OS & OSX_OS)
#    include <OpenCL/OpenCl.h>
#  else
#    include "CL/opencl.h"
#  endif
#endif

#if (OCCA_CUDA_ENABLED)
#  include <cuda.h>
#endif

#if (OCCA_HSA_ENABLED)
#  if   (OCCA_OS & LINUX_OS)
#  elif (OCCA_OS & OSX_OS)
#  else
#  endif
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
  class deviceInfo;
  class kernelDatabase;

  //---[ Typedefs ]-----------------------
  typedef std::vector<int>          intVector_t;
  typedef std::vector<intVector_t>  intVecVector_t;
  typedef std::vector<std::string>  stringVector_t;
  //======================================


  //---[ Globals & Flags ]------------------------
  extern const int parserVersion;

  extern kernelInfo defaultKernelInfo;

  extern const int autoDetect;
  extern const int srcInUva, destInUva;

  extern bool uvaEnabledByDefault_f;
  extern bool verboseCompilation_f;

  void setVerboseCompilation(const bool value);

  namespace flags {
    extern const int checkCacheDir;
  }

  bool hasSerialEnabled();
  bool hasPthreadsEnabled();
  bool hasOpenMPEnabled();
  bool hasOpenCLEnabled();
  bool hasCUDAEnabled();
  bool hasCOIEnabled();
  bool hasHSAEnabled();
  //==============================================


  //---[ Typedefs ]-------------------------------
  typedef void* stream_t;

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
    if(type & Altera) return "Altera";

    return "N/A";
  }

  static const bool useParser = true;

  static const int usingOKL    = (1 << 0);
  static const int usingOFL    = (1 << 1);
  static const int usingNative = (1 << 2);

  static const int useLoopy  = (1 << 0);
  static const int useFloopy = (1 << 1);

  typedef void (*handleFunction_t)(const int *occaKernelInfoArgs,
                                   int occaInnerId0,
                                   int occaInnerId1,
                                   int occaInnerId2, ...);
  //==============================================

  //---[ Mode ]-----------------------------------
  static const occa::mode NoMode   = 0;
  static const occa::mode Serial   = (1 << 20);
  static const occa::mode OpenMP   = (1 << 21);
  static const occa::mode OpenCL   = (1 << 22);
  static const occa::mode CUDA     = (1 << 23);
  static const occa::mode HSA      = (1 << 24);
  static const occa::mode Pthreads = (1 << 25);
  static const occa::mode COI      = (1 << 26);

  static const int onChipMode = (Serial |
                                 OpenMP |
                                 Pthreads);

  static const int offChipMode = (OpenCL |
                                  CUDA   |
                                  HSA);

  static const occa::mode SerialIndex   = 0;
  static const occa::mode OpenMPIndex   = 1;
  static const occa::mode OpenCLIndex   = 2;
  static const occa::mode CUDAIndex     = 3;
  static const occa::mode HSAIndex      = 4;
  static const occa::mode PthreadsIndex = 5;
  static const occa::mode COIIndex      = 6;
  static const int modeCount = 7;

  inline std::string modeToStr(const occa::mode &m){
    if(m & Serial)   return "Serial";
    if(m & OpenMP)   return "OpenMP";
    if(m & OpenCL)   return "OpenCL";
    if(m & CUDA)     return "CUDA";
    if(m & HSA)      return "HSA";
    if(m & Pthreads) return "Pthreads";
    if(m & COI)      return "COI";

    OCCA_CHECK(false, "Mode [" << m << "] is not valid");

    return "No mode";
  }

  inline mode strToMode(const std::string &str){
    const std::string upStr = upString(str);

    if(upStr == "SERIAL")   return Serial;
    if(upStr == "OPENMP")   return OpenMP;
    if(upStr == "OPENCL")   return OpenCL;
    if(upStr == "CUDA")     return CUDA;
    if(upStr == "HSA")      return HSA;
    if(upStr == "PTHREADS") return Pthreads;
    if(upStr == "COI")      return COI;

    OCCA_CHECK(false, "Mode [" << str << "] is not valid");

    return Serial;
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

    if(info_ & Serial)   ret += std::string(count++ ? ", " : "") + "Serial";
    if(info_ & OpenMP)   ret += std::string(count++ ? ", " : "") + "OpenMP";
    if(info_ & OpenCL)   ret += std::string(count++ ? ", " : "") + "OpenCL";
    if(info_ & CUDA)     ret += std::string(count++ ? ", " : "") + "CUDA";
    if(info_ & HSA)      ret += std::string(count++ ? ", " : "") + "HSA";
    if(info_ & Pthreads) ret += std::string(count++ ? ", " : "") + "Pthreads";
    if(info_ & COI)      ret += std::string(count++ ? ", " : "") + "COI";

    if(count)
      return ret;
    else
      return "N/A";
  }
  //==============================================


  //---[ Helper Classes ]-------------------------
  class argInfoMap {
  public:
    std::map<std::string, std::string> iMap;

    argInfoMap();

    argInfoMap(const std::string &infos);

    inline bool has(const std::string &info){
      return (iMap.find(info) != iMap.end());
    }

    inline void remove(const std::string &info){
      std::map<std::string, std::string>::iterator it = iMap.find(info);

      if(it != iMap.end())
        iMap.erase(it);
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

    friend std::ostream& operator << (std::ostream &out, const argInfoMap &m);
  };

  template <>
  inline void argInfoMap::set(const std::string &info, const std::string &value){
    iMap[info] = value;
  }

  class dim {
  public:
    uintptr_t x, y, z;

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
    inline uintptr_t  operator [] (int i) const;
  };

  union kernelArgData_t {
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

  namespace kArgInfo {
    static const char none       = 0;
    static const char usePointer = (1 << 0);
    static const char hasTexture = (1 << 1);
  }

  class kernelArg_t {
  public:
    occa::device_v *dHandle;
    occa::memory_v *mHandle;

    kernelArgData_t data;
    uintptr_t       size;
    char            info;

    inline kernelArg_t(){
      dHandle = NULL;
      mHandle = NULL;

      ::memset(&data, 0, sizeof(data));
      size = 0;
      info = kArgInfo::none;
    }

    inline kernelArg_t(const kernelArg_t &k){
      *this = k;
    }

    inline kernelArg_t& operator = (const kernelArg_t &k){
      dHandle = k.dHandle;
      mHandle = k.mHandle;

      ::memcpy(&data, &(k.data), sizeof(data));
      size = k.size;
      info = k.info;

      return *this;
    }

    inline ~kernelArg_t(){}

    inline void* ptr() const {
      return ((info & kArgInfo::usePointer) ? data.void_ : (void*) &data);
    }
  };

  class kernelArg {
  public:
    int argc;
    kernelArg_t args[2];

    inline kernelArg(){
      argc = 0;
    }

    inline ~kernelArg(){}

    inline kernelArg(kernelArg_t &arg_){
      argc = 1;

      args[0] = arg_;
    }

    inline kernelArg(const kernelArg &k){
      argc = k.argc;

      args[0] = k.args[0];
      args[1] = k.args[1];
    }

    inline kernelArg& operator = (const kernelArg &k){
      argc = k.argc;

      args[0] = k.args[0];
      args[1] = k.args[1];

      return *this;
    }

    template <class TM>
    inline kernelArg(const TM &arg_){
      argc = 1;

      args[0].data.void_ = const_cast<TM*>(&arg_);
      args[0].size       = sizeof(TM);
      args[0].info       = kArgInfo::usePointer;
    }

    template <class TM> inline kernelArg(TM *arg_);
    template <class TM> inline kernelArg(const TM *carg_);

    inline occa::device getDevice() const;

    inline void setupForKernelCall(const bool isConst) const;
  };

  OCCA_KERNEL_ARG_CONSTRUCTOR(int)
  OCCA_KERNEL_ARG_CONSTRUCTOR(char)
  OCCA_KERNEL_ARG_CONSTRUCTOR(short)
  OCCA_KERNEL_ARG_CONSTRUCTOR(long)

  OCCA_KERNEL_ARG_CONSTRUCTOR_ALIAS(unsigned int  , uint)
  OCCA_KERNEL_ARG_CONSTRUCTOR_ALIAS(unsigned char , uchar)
  OCCA_KERNEL_ARG_CONSTRUCTOR_ALIAS(unsigned short, ushort)

  OCCA_KERNEL_ARG_CONSTRUCTOR(float)
  OCCA_KERNEL_ARG_CONSTRUCTOR(double)
  // 32 bit: uintptr_t == unsigned int
#if OCCA_64_BIT
  OCCA_KERNEL_ARG_CONSTRUCTOR(uintptr_t)
#endif

  class streamTag {
  public:
    double tagTime;
    void *handle;

#if OCCA_OPENCL_ENABLED
    inline cl_event& clEvent(){
      return (cl_event&) handle;
    }

    inline cl_event& clEvent() const {
      return const_cast<cl_event&>( (const cl_event&) handle );
    }
#endif

#if OCCA_CUDA_ENABLED
    inline CUevent& cuEvent(){
      return (CUevent&) handle;
    }

    inline CUevent& cuEvent() const {
      return const_cast<CUevent&>( (const CUevent&) handle );
    }
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
  //==============================================


  //---[ Kernel ]---------------------------------
  class kernel_v {
    template <occa::mode> friend class occa::kernel_t;
    template <occa::mode> friend class occa::device_t;
    friend class occa::kernel;
    friend class occa::device;

  private:
    std::string strMode;

    void* data;
    occa::device_v *dHandle;

    std::string name;

    parsedKernelInfo metaInfo;

    uintptr_t maximumInnerDimSize_;
    int preferredDimSize_;

    int dims;
    dim inner, outer;

    int nestedKernelCount;
    kernel *nestedKernels;

    int argumentCount;
    kernelArg_t arguments[OCCA_MAX_ARGS];

  public:
    virtual occa::mode mode() = 0;

    virtual inline ~kernel_v(){}

    virtual void* getKernelHandle() = 0;
    virtual void* getProgramHandle() = 0;

    virtual std::string fixBinaryName(const std::string &filename) = 0;

    virtual kernel_v* buildFromSource(const std::string &filename,
                                      const std::string &functionName,
                                      const kernelInfo &info_ = defaultKernelInfo) = 0;

    virtual kernel_v* buildFromBinary(const std::string &filename,
                                      const std::string &functionName) = 0;

    virtual kernel_v* loadFromLibrary(const char *cache,
                                      const std::string &functionName) = 0;

    virtual uintptr_t maximumInnerDimSize() = 0;
    virtual int preferredDimSize() = 0;

#include "occa/operators/virtualDeclarations.hpp"

    virtual void free() = 0;
  };

  template <occa::mode mode_>
  class kernel_t : public kernel_v {
  public:
    inline occa::mode mode(){
      return mode_;
    }

    kernel_t();
    kernel_t(const std::string &filename,
             const std::string &functionName,
             const kernelInfo &info_ = defaultKernelInfo);

    kernel_t(const kernel_t<mode_> &k);
    kernel_t<mode_>& operator = (const kernel_t<mode_> &k);

    ~kernel_t();

    void* getKernelHandle();
    void* getProgramHandle();

    std::string fixBinaryName(const std::string &filename);

    kernel_t<mode_>* buildFromSource(const std::string &filename,
                                     const std::string &functionName,
                                     const kernelInfo &info_ = defaultKernelInfo);

    kernel_t<mode_>* buildFromBinary(const std::string &filename,
                                     const std::string &functionName);

    kernel_t<mode_>* loadFromLibrary(const char *cache,
                                     const std::string &functionName);

    uintptr_t maximumInnerDimSize();
    int preferredDimSize();

#include "occa/operators/declarations.hpp"

    void free();
  };

  class kernel {
    friend class occa::device;

  private:
    kernel_v *kHandle;

  public:
    kernel();
    kernel(kernel_v *kHandle_);

    kernel(const kernel &k);
    kernel& operator = (const kernel &k);

    inline void checkIfInitialized() const {
      OCCA_CHECK(kHandle != NULL,
                 "Kernel is not initialized");
    }

    void* getKernelHandle();
    void* getProgramHandle();

    kernel_v* getKHandle();

    const std::string& mode();
    const std::string& name();

    occa::device getDevice();

    uintptr_t maximumInnerDimSize();
    int preferredDimSize();

    void setWorkingDims(int dims, dim inner, dim outer);

    void clearArgumentList();

    void addArgument(const int argPos,
                     const kernelArg &arg);

    void runFromArguments();

#include "occa/operators/declarations.hpp"

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
    void addKernel(device_v *d, kernel k);
    void addKernel(const int id, kernel k);

    void loadKernelFromLibrary(device_v *d);

    kernel& operator [] (device d);
    kernel& operator [] (device_v *d);

#include "occa/operators/declarations.hpp"
  };
  //==============================================


  //---[ Memory ]---------------------------------
  void memcpy(void *dest, void *src,
              const uintptr_t bytes,
              const int flags,
              const bool isAsync);

  void memcpy(void *dest, void *src,
              const uintptr_t bytes,
              const int flags = occa::autoDetect);

  void asyncMemcpy(void *dest, void *src,
                   const uintptr_t bytes,
                   const int flags = occa::autoDetect);

  void memcpy(memory dest,
              const void *src,
              const uintptr_t bytes = 0,
              const uintptr_t offset = 0);

  void memcpy(void *dest,
              memory src,
              const uintptr_t bytes = 0,
              const uintptr_t offset = 0);

  void memcpy(memory dest,
              memory src,
              const uintptr_t bytes = 0,
              const uintptr_t destOffset = 0,
              const uintptr_t srcOffset = 0);

  void asyncMemcpy(memory dest,
                   const void *src,
                   const uintptr_t bytes = 0,
                   const uintptr_t offset = 0);

  void asyncMemcpy(void *dest,
                   memory src,
                   const uintptr_t bytes = 0,
                   const uintptr_t offset = 0);

  void asyncMemcpy(memory dest,
                   memory src,
                   const uintptr_t bytes = 0,
                   const uintptr_t destOffset = 0,
                   const uintptr_t srcOffset = 0);

  //---[ Typedefs ]---------------------
  namespace memFlag {
    static const int none         = 0;
    static const int isATexture   = (1 << 0);
    static const int isManaged    = (1 << 1);
    static const int isMapped     = (1 << 2);
    static const int isAWrapper   = (1 << 3);
  }

  namespace uvaFlag {
    static const int inDevice     = (1 << 4);
    static const int leftInDevice = (1 << 5);
    static const int isDirty      = (1 << 6);
  }
  //====================================

  class memory_v {
    template <occa::mode> friend class occa::memory_t;
    template <occa::mode> friend class occa::device_t;
    friend class occa::memory;
    friend class occa::device;
    friend class occa::kernelArg;

  private:
    std::string strMode;

    int memInfo;

    void *handle, *mappedPtr, *uvaPtr;
    occa::device_v *dHandle;

    uintptr_t size;

    occa::textureInfo_t textureInfo;

  public:
    virtual inline occa::mode mode(){ return 0; }
    virtual inline ~memory_v(){}

    inline bool isATexture() const {
      return (memInfo & memFlag::isATexture);
    }

    inline bool isManaged() const {
      return (memInfo & memFlag::isManaged);
    }

    inline bool isMapped() const {
      return (memInfo & memFlag::isMapped);
    }

    inline bool isAWrapper() const {
      return (memInfo & memFlag::isAWrapper);
    }

    inline bool inDevice() const {
      return (memInfo & uvaFlag::inDevice);
    }

    inline bool leftInDevice() const {
      return (memInfo & uvaFlag::leftInDevice);
    }

    inline bool isDirty() const {
      return (memInfo & uvaFlag::isDirty);
    }

    virtual void* getMemoryHandle() = 0;
    virtual void* getTextureHandle() = 0;

    virtual void copyFrom(const void *src,
                          const uintptr_t bytes = 0,
                          const uintptr_t offset = 0) = 0;

    virtual void copyFrom(const memory_v *src,
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

    virtual void asyncCopyFrom(const void *src,
                               const uintptr_t bytes = 0,
                               const uintptr_t offset = 0) = 0;

    virtual void asyncCopyFrom(const memory_v *src,
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

    //---[ Friend Functions ]---------------------

    // Let [memcpy] use private info
    friend void memcpy(void *dest, void *src,
                       const uintptr_t bytes,
                       const int flags);

    friend void asyncMemcpy(void *dest, void *src,
                            const uintptr_t bytes,
                            const int flags);

    friend void memcpy(void *dest, void *src,
                       const uintptr_t bytes,
                       const int flags,
                       const bool isAsync);

    friend void startManaging(void *ptr);
    friend void stopManaging(void *ptr);

    friend void syncToDevice(void *ptr, const uintptr_t bytes);
    friend void syncFromDevice(void *ptr, const uintptr_t bytes);

    friend void syncMemToDevice(occa::memory_v *mem,
                                const uintptr_t bytes,
                                const uintptr_t offset);

    friend void syncMemFromDevice(occa::memory_v *mem,
                                  const uintptr_t bytes,
                                  const uintptr_t offset);

    friend void setupMagicFor(void *ptr);
  };

  template <occa::mode mode_>
  class memory_t : public memory_v {
    friend class occa::device_t<mode_>;

  public:
    memory_t();

    memory_t(const memory_t &m);
    memory_t& operator = (const memory_t &m);

    inline occa::mode mode(){
      return mode_;
    }

    inline ~memory_t(){};

    void* getMemoryHandle();
    void* getTextureHandle();

    void copyFrom(const void *src,
                  const uintptr_t bytes = 0,
                  const uintptr_t offset = 0);

    void copyFrom(const memory_v *src,
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

    void asyncCopyFrom(const void *src,
                       const uintptr_t bytes = 0,
                       const uintptr_t offset = 0);

    void asyncCopyFrom(const memory_v *src,
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
    memory_v *mHandle;

  public:
    memory();
    memory(void *uvaPtr);
    memory(memory_v *mHandle_);

    memory(const memory &m);
    memory& operator = (const memory &m);

    inline void checkIfInitialized() const {
      OCCA_CHECK(mHandle != NULL,
                 "Memory is not initialized");
    }

    memory_v* getMHandle();
    device_v* getDHandle();

    const std::string& mode();

    inline uintptr_t bytes() const {
      if(mHandle == NULL)
        return 0;

      return mHandle->size;
    }

    inline bool isATexture() const {
      return (mHandle->memInfo & memFlag::isATexture);
    }

    inline bool isManaged() const {
      return (mHandle->memInfo & memFlag::isManaged);
    }

    inline bool isMapped() const {
      return (mHandle->memInfo & memFlag::isMapped);
    }

    inline bool isAWrapper() const {
      return (mHandle->memInfo & memFlag::isAWrapper);
    }

    inline bool inDevice() const {
      return (mHandle->memInfo & uvaFlag::inDevice);
    }

    inline bool leftInDevice() const {
      return (mHandle->memInfo & uvaFlag::leftInDevice);
    }

    inline bool isDirty() const {
      return (mHandle->memInfo & uvaFlag::isDirty);
    }

    void* textureArg1() const;
    void* textureArg2() const;

    void* getMappedPointer();
    void* getMemoryHandle();
    void* getTextureHandle();

    void placeInUva();
    void manage();

    void syncToDevice(const uintptr_t bytes, const uintptr_t offset);
    void syncFromDevice(const uintptr_t bytes, const uintptr_t offset);

    bool uvaIsDirty();
    void uvaMarkDirty();
    void uvaMarkClean();

    void copyFrom(const void *src,
                  const uintptr_t bytes = 0,
                  const uintptr_t offset = 0);

    void copyFrom(const memory src,
                  const uintptr_t bytes = 0,
                  const uintptr_t destOffset = 0,
                  const uintptr_t srcOffset = 0);

    void copyTo(void *dest,
                const uintptr_t bytes = 0,
                const uintptr_t offset = 0);

    void copyTo(memory dest,
                const uintptr_t bytes = 0,
                const uintptr_t destOffset = 0,
                const uintptr_t srcOffset = 0);

    void asyncCopyFrom(const void *src,
                       const uintptr_t bytes = 0,
                       const uintptr_t offset = 0);

    void asyncCopyFrom(const memory src,
                       const uintptr_t bytes = 0,
                       const uintptr_t destOffset = 0,
                       const uintptr_t srcOffset = 0);

    void asyncCopyTo(void *dest,
                     const uintptr_t bytes = 0,
                     const uintptr_t offset = 0);

    void asyncCopyTo(memory dest,
                     const uintptr_t bytes = 0,
                     const uintptr_t destOffset = 0,
                     const uintptr_t srcOffset = 0);

    void free();
  };


  //---[ Device ]---------------------------------
  void printAvailableDevices();

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

  namespace cl {
    occa::device wrapDevice(void *platformIDPtr,
                            void *deviceIDPtr,
                            void *contextPtr);

#if OCCA_OPENCL_ENABLED
    occa::device wrapDevice(cl_platform_id platformID,
                            cl_device_id deviceID,
                            cl_context context);
#endif
  }

  namespace cuda {
    occa::device wrapDevice(void *devicePtr,
                            void *contextPtr);

#if OCCA_CUDA_ENABLED
    occa::device wrapDevice(CUdevice device,
                            CUcontext context);
#endif
  }

#if OCCA_HSA_ENABLED
  namespace hsa {
    occa::device wrapDevice();
  }
#endif

  namespace coi {
    occa::device wrapDevice(void *coiDevicePtr);

#if OCCA_COI_ENABLED
    occa::device wrapDevice(COIENGINE coiDevice);
#endif
  }

  class device_v {
    template <occa::mode> friend class occa::kernel_t;
    template <occa::mode> friend class occa::memory_t;
    template <occa::mode> friend class occa::device_t;
    friend class occa::kernel;
    friend class occa::memory;
    friend class occa::device;
    friend class occa::kernelDatabase;

  private:
    std::string strMode;

    int modelID_, id_;

    void* data;

    std::string compiler, compilerEnvScript, compilerFlags;

    bool uvaEnabled_;
    ptrRangeMap_t uvaMap;
    memoryVector_t uvaDirtyMemory;

    stream_t currentStream;
    std::vector<stream_t> streams;

    uintptr_t bytesAllocated;

    int simdWidth_;

  public:
    virtual occa::mode mode() = 0;

    virtual int id() = 0;
    virtual int modelID() = 0;

    virtual inline ~device_v(){}

    virtual void setup(argInfoMap &aim) = 0;

    virtual void* getContextHandle() = 0;

    virtual void addOccaHeadersToInfo(kernelInfo &info) = 0;
    virtual std::string getInfoSalt(const kernelInfo &info) = 0;
    virtual deviceIdentifier getIdentifier() const = 0;

    virtual void getEnvironmentVariables() = 0;

    virtual void setCompiler(const std::string &compiler_) = 0;
    virtual void setCompilerEnvScript(const std::string &compilerEnvScript_) = 0;
    virtual void setCompilerFlags(const std::string &compilerFlags_) = 0;

    virtual void flush()  = 0;
    virtual void finish() = 0;

    virtual bool fakesUva() = 0;
    virtual bool hasUvaEnabled() = 0;

    virtual void waitFor(streamTag tag) = 0;

    virtual stream_t createStream() = 0;
    virtual void freeStream(stream_t s) = 0;
    virtual stream_t wrapStream(void *handle_) = 0;

    virtual streamTag tagStream() = 0;
    virtual double timeBetween(const streamTag &startTag, const streamTag &endTag) = 0;

    virtual std::string fixBinaryName(const std::string &filename) = 0;

    virtual kernel_v* buildKernelFromSource(const std::string &filename,
                                            const std::string &functionName,
                                            const kernelInfo &info_ = defaultKernelInfo) = 0;

    virtual kernel_v* buildKernelFromBinary(const std::string &filename,
                                            const std::string &functionName) = 0;

    virtual void cacheKernelInLibrary(const std::string &filename,
                                      const std::string &functionName,
                                      const kernelInfo &info_ = defaultKernelInfo) = 0;

    virtual kernel_v* loadKernelFromLibrary(const char *cache,
                                            const std::string &functionName) = 0;

#if OCCA_OPENCL_ENABLED
    friend occa::device cl::wrapDevice(cl_platform_id platformID,
                                       cl_device_id deviceID,
                                       cl_context context);
#endif

#if OCCA_CUDA_ENABLED
    friend occa::device cuda::wrapDevice(CUdevice device, CUcontext context);
#endif

#if OCCA_HSA_ENABLED
    friend occa::device hsa::wrapDevice(CUdevice device, CUcontext context);
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
                             void* src) = 0;

    virtual memory_v* textureAlloc(const int dim, const occa::dim &dims,
                                   void *src,
                                   occa::formatType type, const int permissions) = 0;

    virtual memory_v* mappedAlloc(const uintptr_t bytes,
                                  void *src) = 0;

    virtual void free() = 0;

    virtual uintptr_t memorySize() = 0;
    virtual int simdWidth() = 0;
  };

  template <occa::mode mode_>
  class device_t : public device_v {
    template <occa::mode> friend class occa::kernel_t;

  public:
    inline occa::mode mode(){
      return mode_;
    }

    inline int id(){
      return id_;
    }

    inline int modelID(){
      return modelID_;
    }

    device_t();
    device_t(const int arg1, const int arg2);

    inline ~device_t(){}

    device_t(const device_t<mode_> &k);
    device_t<mode_>& operator = (const device_t<mode_> &k);

    void setup(argInfoMap &aim);

    void* getContextHandle();

    void addOccaHeadersToInfo(kernelInfo &info);
    std::string getInfoSalt(const kernelInfo &info);
    deviceIdentifier getIdentifier() const;

    void getEnvironmentVariables();

    static void appendAvailableDevices(std::vector<device> &dList);

    void setCompiler(const std::string &compiler_);
    void setCompilerEnvScript(const std::string &compilerEnvScript_);
    void setCompilerFlags(const std::string &compilerFlags_);

    void flush();
    void finish();

    bool fakesUva();

    inline bool hasUvaEnabled(){
      return uvaEnabled_;
    }

    void waitFor(streamTag tag);

    stream_t createStream();
    void freeStream(stream_t s);
    stream_t wrapStream(void *handle_);

    streamTag tagStream();
    double timeBetween(const streamTag &startTag, const streamTag &endTag);

    std::string fixBinaryName(const std::string &filename);

    kernel_v* buildKernelFromSource(const std::string &filename,
                                    const std::string &functionName,
                                    const kernelInfo &info_ = defaultKernelInfo);

    kernel_v* buildKernelFromBinary(const std::string &filename,
                                    const std::string &functionName);

    void cacheKernelInLibrary(const std::string &filename,
                              const std::string &functionName,
                              const kernelInfo &info_ = defaultKernelInfo);

    kernel_v* loadKernelFromLibrary(const char *cache,
                                    const std::string &functionName);

#if OCCA_OPENCL_ENABLED
    friend occa::device cl::wrapDevice(cl_platform_id platformID,
                                       cl_device_id deviceID,
                                       cl_context context);
#endif

#if OCCA_CUDA_ENABLED
    friend occa::device cuda::wrapDevice(CUdevice device, CUcontext context);
#endif

#if OCCA_HSA_ENABLED
    friend occa::device hsa::wrapDevice(CUdevice device, CUcontext context);
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
                     void *src);

    memory_v* textureAlloc(const int dim, const occa::dim &dims,
                           void *src,
                           occa::formatType type, const int permissions);

    memory_v* mappedAlloc(const uintptr_t bytes,
                          void *src);

    void free();

    uintptr_t memorySize();
    int simdWidth();
  };

  class stream {
  public:
    device_v *dHandle;
    stream_t handle;

    inline stream() :
      dHandle(NULL),
      handle(NULL) {}

    inline stream(device_v *dHandle_, stream_t handle_) :
      dHandle(dHandle_),
      handle(handle_) {}

    inline stream(const stream &s) :
      dHandle(s.dHandle),
      handle(s.handle) {}

    inline stream& operator = (const stream &s){
      dHandle = s.dHandle;
      handle  = s.handle;

      return *this;
    }

    inline void* getStreamHandle(){
      return handle;
    }

    void free();
  };

  class device {
    template <occa::mode> friend class occa::kernel_t;
    template <occa::mode> friend class occa::memory_t;
    template <occa::mode> friend class occa::device_t;

    friend class occa::memory;
    friend class occa::kernelDatabase;

  private:
    device_v *dHandle;

  public:
    device();
    device(device_v *dHandle_);

    device(deviceInfo &dInfo);
    device(const std::string &infos);

    device(const device &d);
    device& operator = (const device &d);

    inline void checkIfInitialized() const {
      OCCA_CHECK(dHandle != NULL,
                 "Device is not initialized");
    }

    void* getContextHandle();
    device_v* getDHandle();

    void setupHandle(occa::mode m);
    void setupHandle(const std::string &m);

    void setup(deviceInfo &dInfo);
    void setup(const std::string &infos);

    void setup(occa::mode m,
               const int arg1, const int arg2);

    void setup(const std::string &m,
               const int arg1, const int arg2);

    uintptr_t memorySize() const;
    uintptr_t memoryAllocated() const;
    // Old name for [memoryAllocated()]
    uintptr_t bytesAllocated() const;

    inline bool hasUvaEnabled(){
      checkIfInitialized();

      return dHandle->hasUvaEnabled();
    }

    deviceIdentifier getIdentifier() const;

    int modelID();
    int id();

    int modeID();
    const std::string& mode();

    void setCompiler(const std::string &compiler_);
    void setCompilerEnvScript(const std::string &compilerEnvScript_);
    void setCompilerFlags(const std::string &compilerFlags_);

    std::string& getCompiler();
    std::string& getCompilerEnvScript();
    std::string& getCompilerFlags();

    void flush();
    void finish();

    void waitFor(streamTag tag);

    stream createStream();
    stream getStream();
    void setStream(stream s);
    stream wrapStream(void *handle_);

    streamTag tagStream();
    double timeBetween(const streamTag &startTag, const streamTag &endTag);

    kernel buildKernel(const std::string &str,
                       const std::string &functionName,
                       const kernelInfo &info_ = defaultKernelInfo);

    kernel buildKernelFromString(const std::string &content,
                                 const std::string &functionName,
                                 const int language = usingOKL);

    kernel buildKernelFromString(const std::string &content,
                                 const std::string &functionName,
                                 const kernelInfo &info_ = defaultKernelInfo,
                                 const int language = usingOKL);

    kernel buildKernelFromSource(const std::string &filename,
                                 const std::string &functionName,
                                 const kernelInfo &info_ = defaultKernelInfo);

    kernel buildKernelFromBinary(const std::string &filename,
                                 const std::string &functionName);

    void cacheKernelInLibrary(const std::string &filename,
                              const std::string &functionName,
                              const kernelInfo &info_ = defaultKernelInfo);

    kernel loadKernelFromLibrary(const char *cache,
                                 const std::string &functionName);

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

#if OCCA_HSA_ENABLED
    friend occa::device hsa::wrapDevice(CUdevice device, CUcontext context);
#endif

#if OCCA_COI_ENABLED
    friend occa::device coi::wrapDevice(COIENGINE coiDevice);
#endif

    memory wrapMemory(void *handle_,
                      const uintptr_t bytes);

    void wrapManagedMemory(void *handle_,
                           const uintptr_t bytes);

    memory wrapTexture(void *handle_,
                       const int dim, const occa::dim &dims,
                       occa::formatType type, const int permissions);

    void wrapManagedTexture(void *handle_,
                            const int dim, const occa::dim &dims,
                            occa::formatType type, const int permissions);

    memory malloc(const uintptr_t bytes,
                  void *src = NULL);

    void* managedAlloc(const uintptr_t bytes,
                       void *src = NULL);

    memory textureAlloc(const int dim, const occa::dim &dims,
                        void *src,
                        occa::formatType type, const int permissions = readWrite);

    void* managedTextureAlloc(const int dim, const occa::dim &dims,
                              void *src,
                              occa::formatType type, const int permissions = readWrite);

    memory mappedAlloc(const uintptr_t bytes,
                       void *src = NULL);

    void* managedMappedAlloc(const uintptr_t bytes,
                             void *src = NULL);

    void freeStream(stream s);

    void free();

    int simdWidth();
  };

  //   ---[ Device Functions ]----------
  extern device host;
  extern device currentDevice;

  void setDevice(device d);
  void setDevice(const std::string &infos);

  device getCurrentDevice();

  extern mutex_t deviceListMutex;
  extern std::vector<device> deviceList;

  std::vector<device>& getDeviceList();

  void setCompiler(const std::string &compiler_);
  void setCompilerEnvScript(const std::string &compilerEnvScript_);
  void setCompilerFlags(const std::string &compilerFlags_);

  std::string& getCompiler();
  std::string& getCompilerEnvScript();
  std::string& getCompilerFlags();

  void flush();
  void finish();

  void waitFor(streamTag tag);

  stream createStream();
  stream getStream();
  void setStream(stream s);
  stream wrapStream(void *handle_);

  streamTag tagStream();

  //   ---[ Kernel Functions ]----------
  kernel buildKernel(const std::string &str,
                     const std::string &functionName,
                     const kernelInfo &info_ = defaultKernelInfo);

  kernel buildKernelFromString(const std::string &content,
                               const std::string &functionName,
                               const int language = usingOKL);

  kernel buildKernelFromString(const std::string &content,
                               const std::string &functionName,
                               const kernelInfo &info_ = defaultKernelInfo,
                               const int language = usingOKL);

  kernel buildKernelFromSource(const std::string &filename,
                               const std::string &functionName,
                               const kernelInfo &info_ = defaultKernelInfo);

  kernel buildKernelFromBinary(const std::string &filename,
                               const std::string &functionName);

  void cacheKernelInLibrary(const std::string &filename,
                            const std::string &functionName,
                            const kernelInfo &info_ = defaultKernelInfo);

  kernel loadKernelFromLibrary(const char *cache,
                               const std::string &functionName);

  kernel buildKernelFromLoopy(const std::string &filename,
                              const std::string &functionName,
                              const int loopyOrFloopy);

  kernel buildKernelFromLoopy(const std::string &filename,
                              const std::string &functionName,
                              const kernelInfo &info_ = defaultKernelInfo,
                              const int loopyOrFloopy = occa::useLoopy);

  //   ---[ Memory Functions ]----------
  memory wrapMemory(void *handle_,
                    const uintptr_t bytes);

  void wrapManagedMemory(void *handle_,
                         const uintptr_t bytes);

  memory wrapTexture(void *handle_,
                     const int dim, const occa::dim &dims,
                     occa::formatType type, const int permissions);

  void wrapManagedTexture(void *handle_,
                          const int dim, const occa::dim &dims,
                          occa::formatType type, const int permissions);

  memory malloc(const uintptr_t bytes,
                void *src = NULL);

  void* managedAlloc(const uintptr_t bytes,
                     void *src = NULL);

  memory textureAlloc(const int dim, const occa::dim &dims,
                      void *src,
                      occa::formatType type, const int permissions = readWrite);

  void* managedTextureAlloc(const int dim, const occa::dim &dims,
                            void *src,
                            occa::formatType type, const int permissions = readWrite);

  memory mappedAlloc(const uintptr_t bytes,
                     void *src = NULL);

  void* managedMappedAlloc(const uintptr_t bytes,
                           void *src = NULL);
  //   =================================

  //   ---[ Free Functions ]------------
  void free(device d);
  void free(stream s);
  void free(kernel k);
  void free(memory m);
  //   =================================

  //---[ KernelArg ]------------------------------
  template <class TM>
  inline kernelArg::kernelArg(TM *arg_){
    ptrRangeMap_t::iterator it = uvaMap.find(arg_);

    if(it != uvaMap.end()){
      occa::memory_v *mHandle = it->second;

      argc = 1;

      args[0].mHandle = mHandle;
      args[0].dHandle = mHandle->dHandle;

      args[0].data.void_ = mHandle->handle;
      args[0].size       = sizeof(void*);
      args[0].info       = kArgInfo::usePointer;
    }
    else {
      argc = 1;

      args[0].data.void_ = arg_;
      args[0].size       = sizeof(TM*);
      args[0].info       = kArgInfo::usePointer;
    }
  }

  template <class TM>
  inline kernelArg::kernelArg(const TM *carg_){
    TM *arg_ = const_cast<TM*>(carg_);

    ptrRangeMap_t::iterator it = uvaMap.find(arg_);

    if(it != uvaMap.end()){
      occa::memory_v *mHandle = it->second;

      argc = 1;

      args[0].mHandle = mHandle;
      args[0].dHandle = mHandle->dHandle;

      args[0].data.void_ = mHandle->handle;
      args[0].size       = sizeof(void*);
      args[0].info       = kArgInfo::usePointer;
    }
    else {
      argc = 1;

      args[0].data.void_ = arg_;
      args[0].size       = sizeof(TM*);
      args[0].info       = kArgInfo::usePointer;
    }
  }

  template <>
  inline kernelArg::kernelArg(const occa::memory &m){
    if(m.mHandle->dHandle->fakesUva()){
      if(!m.isATexture()){
        argc = 1;

        args[0].mHandle = m.mHandle;
        args[0].dHandle = m.mHandle->dHandle;

        args[0].data.void_ = m.mHandle->handle;
        args[0].size       = sizeof(void*);
        args[0].info       = kArgInfo::usePointer;
      }
      else {
        argc = 2;

        args[0].mHandle = m.mHandle;
        args[0].dHandle = m.mHandle->dHandle;

        args[0].data.void_ = m.textureArg1();
        args[0].size       = sizeof(void*);
        args[0].info       = (kArgInfo::usePointer |
                              kArgInfo::hasTexture);

        args[1].mHandle = args[0].mHandle;
        args[1].dHandle = args[0].dHandle;

        args[1].data.void_ = m.textureArg2();
        args[1].size       = sizeof(void*);
        args[1].info       = kArgInfo::usePointer;
      }
    }
    else{
      argc = 1;

      args[0].data.void_ = m.mHandle->handle;
      args[0].size       = sizeof(void*);
      args[0].info       = kArgInfo::usePointer;
    }
  }

  inline occa::device kernelArg::getDevice() const {
    return occa::device(args[0].dHandle);
  }

  inline void kernelArg::setupForKernelCall(const bool isConst) const {
    occa::memory_v *mHandle = args[0].mHandle;

    if(mHandle                      &&
       mHandle->isManaged()         &&
       !mHandle->leftInDevice()     &&
       mHandle->dHandle->fakesUva() &&
       mHandle->dHandle->hasUvaEnabled()){

      if(!mHandle->inDevice()){
        mHandle->copyFrom(mHandle->uvaPtr);
        mHandle->memInfo |= uvaFlag::inDevice;
      }

      if(!isConst && !mHandle->isDirty()){
        uvaDirtyMemory.push_back(mHandle);
        mHandle->memInfo |= uvaFlag::isDirty;
      }
    }
  }
  //==============================================


  //---[ Kernel Database ]------------------------
  inline kernel& kernelDatabase::operator [] (device d){
    return (*this)[d.dHandle];
  }

  inline kernel& kernelDatabase::operator [] (device_v *d){
    OCCA_CHECK(0 <= d->modelID_                 , "Device [modelID] is not set");
    OCCA_CHECK((d->modelID_ < modelKernelCount) , "Kernel is not compiled for chosen device");
    OCCA_CHECK(modelKernelAvailable[d->modelID_], "Kernel is not compiled for chosen device");
    OCCA_CHECK(0 <= d->id_                      , "Device not set");

    if((d->id_ < kernelCount) && kernelAllocated[d->id_])
      return kernels[d->id_];

    loadKernelFromLibrary(d);

    return kernels[d->id_];
  }

  inline kernel& device::operator [] (kernelDatabase &kdb){
    return kdb[dHandle];
  }
  //==============================================

  class deviceInfo {
  public:
    std::string infos;

    deviceInfo();

    deviceInfo(const deviceInfo &dInfo);
    deviceInfo& operator = (const deviceInfo &dInfo);

    void append(const std::string &key,
                const std::string &value);
  };

  class kernelInfo {
  public:
    occa::mode mode;
    std::string header, flags;

    kernelInfo();

    kernelInfo(const kernelInfo &p);
    kernelInfo& operator = (const kernelInfo &p);

    kernelInfo& operator += (const kernelInfo &p);

    std::string salt() const;

    std::string getModeHeaderFilename() const ;

    static bool isAnOccaDefine(const std::string &name);

    void addIncludeDefine(const std::string &filename);

    void addInclude(const std::string &filename);

    void removeDefine(const std::string &macro);

    template <class TM>
    inline void addDefine(const std::string &macro, const TM &value){
      std::stringstream ss;

      if(isAnOccaDefine(macro))
        ss << "#undef " << macro << "\n";

      ss << "#define " << macro << " " << value << '\n';

      header = ss.str() + header;
    }

    void addSource(const std::string &content);

    void addCompilerFlag(const std::string &f);

    void addCompilerIncludePath(const std::string &path);
  };

  template <> void kernelInfo::addDefine(const std::string &macro, const std::string &value);
  template <> void kernelInfo::addDefine(const std::string &macro, const float &value);
  template <> void kernelInfo::addDefine(const std::string &macro, const double &value);

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
    switch(i){
    case 0 : return x;
    case 1 : return y;
    default: return z;
    }
  }

  inline uintptr_t dim::operator [] (int i) const {
    switch(i){
    case 0 : return x;
    case 1 : return y;
    default: return z;
    }
  }
}

#endif
