#include "occa/base.hpp"
#include "occa/library.hpp"
#include "occa/parser/parser.hpp"

#include "occa/Serial.hpp"
#include "occa/OpenCL.hpp"
#include "occa/CUDA.hpp"

// Use events for timing!

namespace occa {
  //---[ Typedefs ]-------------------------------
  std::string deviceType(int type) {
    if(type & CPU)     return "CPU";
    if(type & GPU)     return "GPU";
    if(type & FPGA)    return "FPGA";
    if(type & XeonPhi) return "Xeon Phi";

    return "N/A";
  }

  std::string vendor(int type) {
    if(type & Intel)  return "Intel";
    if(type & AMD)    return "AMD";
    if(type & NVIDIA) return "NVIDIA";
    if(type & Altera) return "Altera";

    return "N/A";
  }
  //==============================================

  //---[ Mode ]-----------------------------------
  std::string modeToStr(const occa::mode &m) {
    if(m & Serial)   return "Serial";
    if(m & OpenMP)   return "OpenMP";
    if(m & OpenCL)   return "OpenCL";
    if(m & CUDA)     return "CUDA";
    if(m & HSA)      return "HSA";
    if(m & Pthreads) return "Pthreads";

    OCCA_CHECK(false, "Mode [" << m << "] is not valid");

    return "No mode";
  }

  mode strToMode(const std::string &str) {
    const std::string upStr = upString(str);

    if(upStr == "SERIAL")   return Serial;
    if(upStr == "OPENMP")   return OpenMP;
    if(upStr == "OPENCL")   return OpenCL;
    if(upStr == "CUDA")     return CUDA;
    if(upStr == "HSA")      return HSA;
    if(upStr == "PTHREADS") return Pthreads;

    OCCA_CHECK(false, "Mode [" << str << "] is not valid");

    return Serial;
  }

  std::string modes(int info, int preferredMode) {
    std::string ret = "";
    int info_ = info;
    int count = 0;

    if(preferredMode != 0) {
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

    if(count)
      return ret;
    else
      return "N/A";
  }
  //==============================================

  //---[ Globals & Flags ]------------------------
  const int parserVersion = 100;

  kernelInfo defaultKernelInfo;

  const int autoDetect = (1 << 0);
  const int srcInUva   = (1 << 1);
  const int destInUva  = (1 << 2);

  bool uvaEnabledByDefault_f = false;
  bool verboseCompilation_f  = true;

  void setVerboseCompilation(const bool value) {
    verboseCompilation_f = value;
  }

  namespace flags {
    const int checkCacheDir = (1 << 0);
  }

  bool hasSerialEnabled() {
    return true;
  }

  bool hasPthreadsEnabled() {
    return true;
  }

  bool hasOpenMPEnabled() {
#if OCCA_OPENMP_ENABLED
    return true;
#else
    return false;
#endif
  }

  bool hasOpenCLEnabled() {
#if OCCA_OPENCL_ENABLED
    return true;
#else
    return false;
#endif
  }

  bool hasCUDAEnabled() {
#if OCCA_CUDA_ENABLED
    return true;
#else
    return false;
#endif
  }

  bool hasHSAEnabled() {
#if OCCA_HSA_ENABLED
    return true;
#else
    return false;
#endif
  }
  //==============================================

  //---[ Helper Classes ]-------------------------
  bool argInfoMap::has(const std::string &info) {
    return (iMap.find(info) != iMap.end());
  }

  void argInfoMap::remove(const std::string &info) {
    std::map<std::string, std::string>::iterator it = iMap.find(info);

    if(it != iMap.end())
      iMap.erase(it);
  }

  template <>
  void argInfoMap::set(const std::string &info, const std::string &value) {
    iMap[info] = value;
  }

  std::string argInfoMap::get(const std::string &info) {
    std::map<std::string,std::string>::iterator it = iMap.find(info);

    if(it != iMap.end())
      return it->second;

    return "";
  }

  int argInfoMap::iGet(const std::string &info) {
    std::map<std::string,std::string>::iterator it = iMap.find(info);

    if(it != iMap.end())
      return atoi((it->second).c_str());

    return 0;
  }

  void argInfoMap::iGets(const std::string &info, std::vector<int> &entries) {
    std::map<std::string,std::string>::iterator it = iMap.find(info);

    if(it == iMap.end())
      return;

    const char *c = (it->second).c_str();

    while(*c != '\0') {
      skipWhitespace(c);

      if(isANumber(c)) {
        entries.push_back(atoi(c));
        skipNumber(c);
      }
      else
        ++c;
    }
  }

  dim::dim() :
    x(1),
    y(1),
    z(1) {}

  dim::dim(uintptr_t x_) :
    x(x_),
    y(1),
    z(1) {}

  dim::dim(uintptr_t x_, uintptr_t y_) :
    x(x_),
    y(y_),
    z(1) {}

  dim::dim(uintptr_t x_, uintptr_t y_, uintptr_t z_) :
    x(x_),
    y(y_),
    z(z_) {}

  dim::dim(const dim &d) :
    x(d.x),
    y(d.y),
    z(d.z) {}

  dim& dim::operator = (const dim &d) {
    x = d.x;
    y = d.y;
    z = d.z;

    return *this;
  }

  dim dim::operator + (const dim &d) {
    return dim(x + d.x,
               y + d.y,
               z + d.z);
  }

  dim dim::operator - (const dim &d) {
    return dim(x - d.x,
               y - d.y,
               z - d.z);
  }

  dim dim::operator * (const dim &d) {
    return dim(x * d.x,
               y * d.y,
               z * d.z);
  }

  dim dim::operator / (const dim &d) {
    return dim(x / d.x,
               y / d.y,
               z / d.z);
  }

  bool dim::hasNegativeEntries() {
    return ((x & (1 << (sizeof(uintptr_t) - 1))) ||
            (y & (1 << (sizeof(uintptr_t) - 1))) ||
            (z & (1 << (sizeof(uintptr_t) - 1))));
  }

  uintptr_t& dim::operator [] (int i) {
    switch(i) {
    case 0 : return x;
    case 1 : return y;
    default: return z;
    }
  }

  uintptr_t dim::operator [] (int i) const {
    switch(i) {
    case 0 : return x;
    case 1 : return y;
    default: return z;
    }
  }

  kernelArg_t::kernelArg_t() {
    dHandle = NULL;
    mHandle = NULL;

    ::memset(&data, 0, sizeof(data));
    size = 0;
    info = kArgInfo::none;
  }

  kernelArg_t::kernelArg_t(const kernelArg_t &k) {
    *this = k;
  }

  kernelArg_t& kernelArg_t::operator = (const kernelArg_t &k) {
    dHandle = k.dHandle;
    mHandle = k.mHandle;

    ::memcpy(&data, &(k.data), sizeof(data));
    size = k.size;
    info = k.info;

    return *this;
  }

  kernelArg_t::~kernelArg_t() {}

  void* kernelArg_t::ptr() const {
    return ((info & kArgInfo::usePointer) ? data.void_ : (void*) &data);
  }

  kernelArg::kernelArg() {
    argc = 0;
  }

  kernelArg::~kernelArg() {}

  kernelArg::kernelArg(kernelArg_t &arg_) {
    argc = 1;

    args[0] = arg_;
  }

  kernelArg::kernelArg(const kernelArg &k) {
    argc = k.argc;

    args[0] = k.args[0];
    args[1] = k.args[1];
  }

  kernelArg& kernelArg::operator = (const kernelArg &k) {
    argc = k.argc;

    args[0] = k.args[0];
    args[1] = k.args[1];

    return *this;
  }

  template <> kernelArg::kernelArg(const int &arg_) {
    argc = 1; args[0].data.int_ = arg_; args[0].size = sizeof(int);
  }
  template <> kernelArg::kernelArg(const char &arg_) {
    argc = 1; args[0].data.char_ = arg_; args[0].size = sizeof(char);
  }
  template <> kernelArg::kernelArg(const short &arg_) {
    argc = 1; args[0].data.short_ = arg_; args[0].size = sizeof(short);
  }
  template <> kernelArg::kernelArg(const long &arg_) {
    argc = 1; args[0].data.long_ = arg_; args[0].size = sizeof(long);
  }

  template <> kernelArg::kernelArg(const unsigned int &arg_) {
    argc = 1; args[0].data.uint_ = arg_; args[0].size = sizeof(unsigned int);
  }
  template <> kernelArg::kernelArg(const unsigned char &arg_) {
    argc = 1; args[0].data.uchar_ = arg_; args[0].size = sizeof(unsigned char);
  }
  template <> kernelArg::kernelArg(const unsigned short &arg_) {
    argc = 1; args[0].data.ushort_ = arg_; args[0].size = sizeof(unsigned short);
  }

  template <> kernelArg::kernelArg(const float &arg_) {
    argc = 1; args[0].data.float_ = arg_; args[0].size = sizeof(float);
  }
  template <> kernelArg::kernelArg(const double &arg_) {
    argc = 1; args[0].data.double_ = arg_; args[0].size = sizeof(double);
  }

#if OCCA_64_BIT
  // 32 bit: uintptr_t == unsigned int
  template <> kernelArg::kernelArg(const uintptr_t &arg_) {
    argc = 1; args[0].data.uintptr_t_ = arg_; args[0].size = sizeof(uintptr_t);
  }
#endif

  occa::device kernelArg::getDevice() const {
    return occa::device(args[0].dHandle);
  }

  void kernelArg::setupForKernelCall(const bool isConst) const {
    occa::memory_v *mHandle = args[0].mHandle;

    if(mHandle                      &&
       mHandle->isManaged()         &&
       !mHandle->leftInDevice()     &&
       mHandle->dHandle->fakesUva() &&
       mHandle->dHandle->hasUvaEnabled()) {

      if(!mHandle->inDevice()) {
        mHandle->copyFrom(mHandle->uvaPtr);
        mHandle->memInfo |= uvaFlag::inDevice;
      }

      if(!isConst && !mHandle->isDirty()) {
        uvaDirtyMemory.push_back(mHandle);
        mHandle->memInfo |= uvaFlag::isDirty;
      }
    }
  }

  int kernelArg::argumentCount(const int kArgc, const kernelArg *kArgs) {
    int argc = 0;
    for(int i = 0; i < kArgc; ++i){
      argc += kArgs[i].argc;
    }
    return argc;
  }

  const int uint8FormatIndex  = 0;
  const int uint16FormatIndex = 1;
  const int uint32FormatIndex = 2;
  const int int8FormatIndex   = 3;
  const int int16FormatIndex  = 4;
  const int int32FormatIndex  = 5;
  const int halfFormatIndex   = 6;
  const int floatFormatIndex  = 7;

  const int sizeOfFormats[8] = {1, 2, 4,
                                1, 2, 4,
                                2, 4};

  formatType::formatType(const int format__, const int count__) {
    format_ = format__;
    count_  = count__;
  }

  formatType::formatType(const formatType &ft) {
    format_ = ft.format_;
    count_  = ft.count_;
  }

  formatType& formatType::operator = (const formatType &ft) {
    format_ = ft.format_;
    count_  = ft.count_;

    return *this;
  }

  int formatType::count() const {
    return count_;
  }

  size_t formatType::bytes() const {
    return (sizeOfFormats[format_] * count_);
  }

  const int readOnly  = 1;
  const int readWrite = 2;

  const occa::formatType uint8Format(uint8FormatIndex  , 1);
  const occa::formatType uint8x2Format(uint8FormatIndex, 2);
  const occa::formatType uint8x4Format(uint8FormatIndex, 4);

  const occa::formatType uint16Format(uint16FormatIndex  , 1);
  const occa::formatType uint16x2Format(uint16FormatIndex, 2);
  const occa::formatType uint16x4Format(uint16FormatIndex, 4);

  const occa::formatType uint32Format(uint32FormatIndex  , 1);
  const occa::formatType uint32x2Format(uint32FormatIndex, 2);
  const occa::formatType uint32x4Format(uint32FormatIndex, 4);

  const occa::formatType int8Format(int8FormatIndex  , 1);
  const occa::formatType int8x2Format(int8FormatIndex, 2);
  const occa::formatType int8x4Format(int8FormatIndex, 4);

  const occa::formatType int16Format(int16FormatIndex  , 1);
  const occa::formatType int16x2Format(int16FormatIndex, 2);
  const occa::formatType int16x4Format(int16FormatIndex, 4);

  const occa::formatType int32Format(int32FormatIndex  , 1);
  const occa::formatType int32x2Format(int32FormatIndex, 2);
  const occa::formatType int32x4Format(int32FormatIndex, 4);

  const occa::formatType halfFormat(halfFormatIndex  , 1);
  const occa::formatType halfx2Format(halfFormatIndex, 2);
  const occa::formatType halfx4Format(halfFormatIndex, 4);

  const occa::formatType floatFormat(floatFormatIndex  , 1);
  const occa::formatType floatx2Format(floatFormatIndex, 2);
  const occa::formatType floatx4Format(floatFormatIndex, 4);

  //  |---[ Arg Info Map ]------------------------
  argInfoMap::argInfoMap() {}

  argInfoMap::argInfoMap(const std::string &infos) {
    if(infos.size() == 0)
      return;

    parserNS::expNode expRoot = parserNS::createOrganizedExpNodeFrom(infos);

    parserNS::expNode &csvFlatRoot = *(expRoot.makeCsvFlatHandle());

    for(int i = 0; i < csvFlatRoot.leafCount; ++i) {
      parserNS::expNode &leaf = csvFlatRoot[i];

      std::string &info = (leaf.leafCount ?
                           leaf[0].value  :
                           leaf.value);

      if((info != "mode")        &&
         (info != "UVA")         &&
         (info != "platformID")  &&
         (info != "deviceID")    &&
         (info != "schedule")    &&
         (info != "chunk")       &&
         (info != "threadCount") &&
         (info != "schedule")    &&
         (info != "pinnedCores")) {

        std::cout << "Flag [" << info << "] is not available, skipping it\n";
        continue;
      }

      if(leaf.value != "=") {
        std::cout << "Flag [" << info << "] was not set, skipping it\n";
        continue;
      }

      iMap[info] = leaf[1].toString();
    }

    parserNS::expNode::freeFlatHandle(csvFlatRoot);
  }

  argInfoMap::argInfoMap(argInfoMap &aim) {
    *this = aim;
  }

  argInfoMap& argInfoMap::operator = (argInfoMap &aim) {
    iMap = aim.iMap;
    return *this;
  }

  std::ostream& operator << (std::ostream &out, const argInfoMap &m) {
    std::map<std::string,std::string>::const_iterator it = m.iMap.begin();

    while(it != m.iMap.end()) {
      out << it->first << " = " << it->second << '\n';
      ++it;
    }

    return out;
  }

  //  |---[ Kernel Info ]-------------------------
  kernelInfo::kernelInfo() :
    mode(NoMode),
    header(""),
    flags("") {}

  kernelInfo::kernelInfo(const kernelInfo &p) :
    mode(p.mode),
    header(p.header),
    flags(p.flags) {}

  kernelInfo& kernelInfo::operator = (const kernelInfo &p) {
    mode   = p.mode;
    header = p.header;
    flags  = p.flags;

    return *this;
  }

  kernelInfo& kernelInfo::operator += (const kernelInfo &p) {
    header += p.header;
    flags  += p.flags;

    return *this;
  }

  std::string kernelInfo::salt() const {
    return (header + flags);
  }

  std::string kernelInfo::getModeHeaderFilename() const {
    if(mode & Serial)   return sys::getFilename("[occa]/defines/Serial.hpp");
    if(mode & OpenMP)   return sys::getFilename("[occa]/defines/OpenMP.hpp");
    if(mode & OpenCL)   return sys::getFilename("[occa]/defines/OpenCL.hpp");
    if(mode & CUDA)     return sys::getFilename("[occa]/defines/CUDA.hpp");
    if(mode & HSA)      return sys::getFilename("[occa]/defines/HSA.hpp");
    if(mode & Pthreads) return sys::getFilename("[occa]/defines/Pthreads.hpp");

    return "";
  }

  bool kernelInfo::isAnOccaDefine(const std::string &name) {
    if((name == "OCCA_USING_CPU") ||
       (name == "OCCA_USING_GPU") ||

       (name == "OCCA_USING_SERIAL")   ||
       (name == "OCCA_USING_OPENMP")   ||
       (name == "OCCA_USING_OPENCL")   ||
       (name == "OCCA_USING_CUDA")     ||
       (name == "OCCA_USING_PTHREADS") ||

       (name == "occaInnerDim0") ||
       (name == "occaInnerDim1") ||
       (name == "occaInnerDim2") ||

       (name == "occaOuterDim0") ||
       (name == "occaOuterDim1") ||
       (name == "occaOuterDim2"))
      return true;

    return false;
  }

  void kernelInfo::addIncludeDefine(const std::string &filename) {
    header += "\n#include \"";
    header += filename;
    header += "\"\n";
  }

  void kernelInfo::addInclude(const std::string &filename) {
    header += '\n';
    header += readFile(filename);
    header += '\n';
  }

  void kernelInfo::removeDefine(const std::string &macro) {
    if(!isAnOccaDefine(macro))
      header += "#undef " + macro + '\n';
  }

  void kernelInfo::addSource(const std::string &content) {
    header += content;
  }

  void kernelInfo::addCompilerFlag(const std::string &f) {
    flags += " " + f;
  }

  void kernelInfo::addCompilerIncludePath(const std::string &path) {
#if (OCCA_OS & (LINUX_OS | OSX_OS))
    flags += " -I \"" + path + "\"";
#else
    flags += " /I \"" + path + "\"";
#endif
  }

  flags_t& kernelInfo::getParserFlags() {
    return parserFlags;
  }

  const flags_t& kernelInfo::getParserFlags() const {
    return parserFlags;
  }

  void kernelInfo::addParserFlag(const std::string &flag,
                                 const std::string &value) {

    parserFlags[flag] = value;
  }

  template <>
  void kernelInfo::addDefine(const std::string &macro, const std::string &value) {
    std::stringstream ss;

    if(isAnOccaDefine(macro))
      ss << "#undef " << macro << "\n";

    // Make sure newlines are followed by escape characters
    std::string value2 = "";
    const int chars = value.size();

    for(int i = 0; i < chars; ++i) {
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
    //  |=========================================

    ss << "#define " << macro << " " << value2 << '\n';

    header = ss.str() + header;
  }

  template <>
  void kernelInfo::addDefine(const std::string &macro, const float &value) {
    std::stringstream ss;

    if(isAnOccaDefine(macro))
      ss << "#undef " << macro << "\n";

    ss << "#define " << macro << ' '
       << std::scientific << std::setprecision(8) << value << "f\n";

    header = ss.str() + header;
  }

  template <>
  void kernelInfo::addDefine(const std::string &macro, const double &value) {
    std::stringstream ss;

    if(isAnOccaDefine(macro))
      ss << "#undef " << macro << "\n";

    ss << "#define " << macro << ' '
       << std::scientific << std::setprecision(16) << value << '\n';

    header = ss.str() + header;
  }

  //  |---[ Device Info ]-------------------------
  deviceInfo::deviceInfo() {}

  deviceInfo::deviceInfo(const deviceInfo &dInfo) :
    infos(dInfo.infos) {}

  deviceInfo& deviceInfo::operator = (const deviceInfo &dInfo) {
    infos = dInfo.infos;

    return *this;
  }

  void deviceInfo::append(const std::string &key,
                          const std::string &value) {
    if(infos.size() != 0)
      infos += ',';

    infos += key;
    infos += '=';
    infos += value;
  }
  //==============================================

  //---[ Kernel ]---------------------------------
  kernel* kernel_v::nestedKernelsPtr() {
    return &(nestedKernels[0]);
  }

  int kernel_v::nestedKernelCount() {
    return (int) nestedKernels.size();
  }

  kernelArg* kernel_v::argumentsPtr() {
    return &(arguments[0]);
  }

  int kernel_v::argumentCount() {
    return (int) arguments.size();
  }

  kernel::kernel() :
    kHandle(NULL) {}

  kernel::kernel(kernel_v *kHandle_) :
    kHandle(kHandle_) {}

  kernel::kernel(const kernel &k) :
    kHandle(k.kHandle) {}

  kernel& kernel::operator = (const kernel &k) {
    kHandle = k.kHandle;
    return *this;
  }

  void kernel::checkIfInitialized() const {
    OCCA_CHECK(kHandle != NULL,
               "Kernel is not initialized");
  }

  void* kernel::getKernelHandle() {
    checkIfInitialized();
    return kHandle->getKernelHandle();
  }

  void* kernel::getProgramHandle() {
    checkIfInitialized();
    return kHandle->getProgramHandle();
  }

  kernel_v* kernel::getKHandle() {
    checkIfInitialized();
    return kHandle;
  }

  const std::string& kernel::mode() {
    checkIfInitialized();
    return kHandle->strMode;
  }

  const std::string& kernel::name() {
    checkIfInitialized();
    return kHandle->name;
  }

  occa::device kernel::getDevice() {
    checkIfInitialized();
    return occa::device(kHandle->dHandle);
  }

  void kernel::setWorkingDims(int dims, occa::dim inner, occa::dim outer) {
    checkIfInitialized();

    for(int i = 0; i < dims; ++i) {
      inner[i] += (inner[i] ? 0 : 1);
      outer[i] += (outer[i] ? 0 : 1);
    }

    for(int i = dims; i < 3; ++i)
      inner[i] = outer[i] = 1;

    if (kHandle->nestedKernelCount()) {
      for(int k = 0; k < kHandle->nestedKernelCount(); ++k)
        kHandle->nestedKernels[k].setWorkingDims(dims, inner, outer);
    } else {
      kHandle->dims  = dims;
      kHandle->inner = inner;
      kHandle->outer = outer;
    }
  }

  uintptr_t kernel::maximumInnerDimSize() {
    checkIfInitialized();
    return kHandle->maximumInnerDimSize();
  }

  int kernel::preferredDimSize() {
    checkIfInitialized();

    if(kHandle->nestedKernelCount())
      return 0;

    return kHandle->preferredDimSize();
  }

  void kernel::clearArgumentList() {
    checkIfInitialized();
    kHandle->arguments.clear();
  }

  void kernel::addArgument(const int argPos,
                           const kernelArg &arg) {
    checkIfInitialized();

    if(kHandle->argumentCount() <= argPos) {
      OCCA_CHECK(argPos < OCCA_MAX_ARGS,
                 "Kernels can only have at most [" << OCCA_MAX_ARGS << "] arguments,"
                 << " [" << argPos << "] arguments were set");

      kHandle->arguments.reserve(argPos + 1);
    }

    kHandle->arguments.insert(kHandle->arguments.begin() + argPos, arg);
  }

  void kernel::runFromArguments() {
    checkIfInitialized();

    // Add nestedKernels
    if (kHandle->nestedKernelCount())
      kHandle->arguments.insert(kHandle->arguments.begin(),
                                kHandle->nestedKernelsPtr());

    kHandle->runFromArguments(kHandle->argumentCount(),
                              kHandle->argumentsPtr());

    // Remove nestedKernels
    if (kHandle->nestedKernelCount())
      kHandle->arguments.erase(kHandle->arguments.begin());
  }

#include "operators/definitions.cpp"

  void kernel::free() {
    checkIfInitialized();

    if(kHandle->nestedKernelCount()) {
      for(int k = 0; k < kHandle->nestedKernelCount(); ++k)
        kHandle->nestedKernels[k].free();
    }

    kHandle->free();

    delete kHandle;
    kHandle = NULL;
  }

  kernelDatabase::kernelDatabase() :
    kernelName(""),
    modelKernelCount(0),
    kernelCount(0) {}

  kernelDatabase::kernelDatabase(const std::string kernelName_) :
    kernelName(kernelName_),
    modelKernelCount(0),
    kernelCount(0) {}

  kernelDatabase::kernelDatabase(const kernelDatabase &kdb) :
    kernelName(kdb.kernelName),

    modelKernelCount(kdb.modelKernelCount),
    modelKernelAvailable(kdb.modelKernelAvailable),

    kernelCount(kdb.kernelCount),
    kernels(kdb.kernels),
    kernelAllocated(kdb.kernelAllocated) {}


  kernelDatabase& kernelDatabase::operator = (const kernelDatabase &kdb) {
    kernelName = kdb.kernelName;

    modelKernelCount     = kdb.modelKernelCount;
    modelKernelAvailable = kdb.modelKernelAvailable;

    kernelCount     = kdb.kernelCount;
    kernels         = kdb.kernels;
    kernelAllocated = kdb.kernelAllocated;

    return *this;
  }

  void kernelDatabase::modelKernelIsAvailable(const int id) {
    OCCA_CHECK(0 <= id,
               "Model kernel for ID [" << id << "] was not found");

    if(modelKernelCount <= id) {
      modelKernelCount = (id + 1);
      modelKernelAvailable.resize(modelKernelCount, false);
    }

    modelKernelAvailable[id] = true;
  }

  void kernelDatabase::addKernel(device d, kernel k) {
    addKernel(d.dHandle->id_, k);
  }

  void kernelDatabase::addKernel(device_v *d, kernel k) {
    addKernel(d->id_, k);
  }

  void kernelDatabase::addKernel(const int id, kernel k) {
    OCCA_CHECK(0 <= id,
               "Model kernel for ID [" << id << "] was not found");

    if(kernelCount <= id) {
      kernelCount = (id + 1);

      kernels.resize(kernelCount);
      kernelAllocated.resize(kernelCount, false);
    }

    kernels[id] = k;
    kernelAllocated[id] = true;
  }

  void kernelDatabase::loadKernelFromLibrary(device_v *d) {
    addKernel(d, library::loadKernel(d, kernelName));
  }
  //==============================================


  //---[ Memory ]---------------------------------
  bool memory_v::isATexture() const {
    return (memInfo & memFlag::isATexture);
  }

  bool memory_v::isManaged() const {
    return (memInfo & memFlag::isManaged);
  }

  bool memory_v::isMapped() const {
    return (memInfo & memFlag::isMapped);
  }

  bool memory_v::isAWrapper() const {
    return (memInfo & memFlag::isAWrapper);
  }

  bool memory_v::inDevice() const {
    return (memInfo & uvaFlag::inDevice);
  }

  bool memory_v::leftInDevice() const {
    return (memInfo & uvaFlag::leftInDevice);
  }

  bool memory_v::isDirty() const {
    return (memInfo & uvaFlag::isDirty);
  }

  memory::memory() :
    mHandle(NULL) {}

  memory::memory(void *uvaPtr) {
    // Default to uvaPtr is actually a memory_v*
    memory_v *mHandle_ = (memory_v*) uvaPtr;

    ptrRangeMap_t::iterator it = uvaMap.find(uvaPtr);

    if(it != uvaMap.end())
      mHandle_ = it->second;

    mHandle = mHandle_;
  }

  memory::memory(memory_v *mHandle_) :
    mHandle(mHandle_) {}

  memory::memory(const memory &m) :
    mHandle(m.mHandle) {}

  memory& memory::swap(memory &m) {
    memory_v *tmp = mHandle;
    mHandle       = m.mHandle;
    m.mHandle     = tmp;

    return *this;
  }

  memory& memory::operator = (const memory &m) {
    mHandle = m.mHandle;
    return *this;
  }

  void memory::checkIfInitialized() const {
    OCCA_CHECK(mHandle != NULL,
               "Memory is not initialized");
  }

  memory_v* memory::getMHandle() {
    checkIfInitialized();
    return mHandle;
  }

  device_v* memory::getDHandle() {
    checkIfInitialized();
    return mHandle->dHandle;
  }

  const std::string& memory::mode() {
    checkIfInitialized();
    return mHandle->strMode;
  }

  uintptr_t memory::bytes() const {
    if(mHandle == NULL)
      return 0;
    return mHandle->size;
  }

  bool memory::isATexture() const {
    return (mHandle->memInfo & memFlag::isATexture);
  }

  bool memory::isManaged() const {
    return (mHandle->memInfo & memFlag::isManaged);
  }

  bool memory::isMapped() const {
    return (mHandle->memInfo & memFlag::isMapped);
  }

  bool memory::isAWrapper() const {
    return (mHandle->memInfo & memFlag::isAWrapper);
  }

  bool memory::inDevice() const {
    return (mHandle->memInfo & uvaFlag::inDevice);
  }

  bool memory::leftInDevice() const {
    return (mHandle->memInfo & uvaFlag::leftInDevice);
  }

  bool memory::isDirty() const {
    return (mHandle->memInfo & uvaFlag::isDirty);
  }

  void* memory::textureArg1() const {
    checkIfInitialized();

#if !OCCA_CUDA_ENABLED
    return (void*) mHandle;
#else
    if(mHandle->mode() != CUDA)
      return (void*) mHandle;
    else
      return &(((CUDATextureData_t*) mHandle->handle)->surface);
#endif
  }

  void* memory::textureArg2() const {
    checkIfInitialized();
    return (void*) ((mHandle->textureInfo).arg);
  }

  void* memory::getMappedPointer() {
    checkIfInitialized();
    return mHandle->mappedPtr;
  }

  void* memory::getMemoryHandle() {
    checkIfInitialized();
    return mHandle->getMemoryHandle();
  }

  void* memory::getTextureHandle() {
    checkIfInitialized();
    return mHandle->getTextureHandle();
  }

  void memory::placeInUva() {
    checkIfInitialized();

    if( !(mHandle->dHandle->fakesUva()) ) {
      mHandle->uvaPtr = mHandle->handle;
    }
    else if(mHandle->isMapped()) {
      mHandle->uvaPtr = mHandle->mappedPtr;
    }
    else{
      mHandle->uvaPtr = cpu::malloc(mHandle->size);
    }

    ptrRange_t uvaRange;

    uvaRange.start = (char*) (mHandle->uvaPtr);
    uvaRange.end   = (uvaRange.start + mHandle->size);

    uvaMap[uvaRange]                   = mHandle;
    mHandle->dHandle->uvaMap[uvaRange] = mHandle;

    // Needed for kernelArg.void_ -> mHandle checks
    if(mHandle->uvaPtr != mHandle->handle)
      uvaMap[mHandle->handle] = mHandle;
  }

  void memory::manage() {
    checkIfInitialized();
    placeInUva();
    mHandle->memInfo |= memFlag::isManaged;
  }

  void memory::syncToDevice(const uintptr_t bytes,
                            const uintptr_t offset) {
    checkIfInitialized();

    if(mHandle->dHandle->fakesUva()) {
      uintptr_t bytes_ = ((bytes == 0) ? mHandle->size : bytes);

      copyTo(mHandle->uvaPtr, bytes_, offset);

      mHandle->memInfo |=  uvaFlag::inDevice;
      mHandle->memInfo &= ~uvaFlag::isDirty;

      removeFromDirtyMap(mHandle);
    }
  }

  void memory::syncFromDevice(const uintptr_t bytes,
                              const uintptr_t offset) {
    checkIfInitialized();

    if(mHandle->dHandle->fakesUva()) {
      uintptr_t bytes_ = ((bytes == 0) ? mHandle->size : bytes);

      copyFrom(mHandle->uvaPtr, bytes_, offset);

      mHandle->memInfo &= ~uvaFlag::inDevice;
      mHandle->memInfo &= ~uvaFlag::isDirty;

      removeFromDirtyMap(mHandle);
    }
  }

  bool memory::uvaIsDirty() {
    checkIfInitialized();
    return (mHandle && mHandle->isDirty());
  }

  void memory::uvaMarkDirty() {
    checkIfInitialized();
    if(mHandle != NULL)
      mHandle->memInfo |= uvaFlag::isDirty;
  }

  void memory::uvaMarkClean() {
    checkIfInitialized();
    if(mHandle != NULL)
      mHandle->memInfo &= ~uvaFlag::isDirty;
  }

  void memory::copyFrom(const void *src,
                        const uintptr_t bytes,
                        const uintptr_t offset) {
    checkIfInitialized();
    mHandle->copyFrom(src, bytes, offset);
  }

  void memory::copyFrom(const memory src,
                        const uintptr_t bytes,
                        const uintptr_t destOffset,
                        const uintptr_t srcOffset) {
    checkIfInitialized();

    if(mHandle->dHandle == src.mHandle->dHandle) {
      mHandle->copyFrom(src.mHandle, bytes, destOffset, srcOffset);
    }
    else{
      memory_v *srcHandle  = src.mHandle;
      memory_v *destHandle = mHandle;

      const occa::mode modeS = srcHandle->mode();
      const occa::mode modeD = destHandle->mode();

      if(modeS & onChipMode) {
        destHandle->copyFrom(srcHandle->getMemoryHandle(),
                             bytes, destOffset);
      }
      else if(modeD & onChipMode) {
        srcHandle->copyTo(destHandle->getMemoryHandle(),
                          bytes, srcOffset);
      }
      else{
        OCCA_CHECK(((modeS == CUDA) && (modeD == CUDA)),
                   "Peer-to-peer is not supported between ["
                   << modeToStr(modeS) << "] and ["
                   << modeToStr(modeD) << "]");

#if OCCA_CUDA_ENABLED
        CUDADeviceData_t &srcDevData  =
          *((CUDADeviceData_t*) srcHandle->dHandle->data);

        CUDADeviceData_t &destDevData =
          *((CUDADeviceData_t*) destHandle->dHandle->data);

        CUdeviceptr srcMem  = *(((CUdeviceptr*) srcHandle->handle)  + srcOffset);
        CUdeviceptr destMem = *(((CUdeviceptr*) destHandle->handle) + destOffset);

        if(!srcDevData.p2pEnabled)
          cuda::enablePeerToPeer(srcDevData.context);

        if(!destDevData.p2pEnabled)
          cuda::enablePeerToPeer(destDevData.context);

        cuda::checkPeerToPeer(destDevData.device,
                              srcDevData.device);

        cuda::peerToPeerMemcpy(destDevData.device,
                               destDevData.context,
                               destMem,

                               srcDevData.device,
                               srcDevData.context,
                               srcMem,

                               bytes,
                               *((CUstream*) srcHandle->dHandle->currentStream));
#endif
      }
    }
  }

  void memory::copyTo(void *dest,
                      const uintptr_t bytes,
                      const uintptr_t offset) {
    checkIfInitialized();
    mHandle->copyTo(dest, bytes, offset);
  }

  void memory::copyTo(memory dest,
                      const uintptr_t bytes,
                      const uintptr_t destOffset,
                      const uintptr_t srcOffset) {
    checkIfInitialized();

    if(mHandle->dHandle == dest.mHandle->dHandle) {
      mHandle->copyTo(dest.mHandle, bytes, destOffset, srcOffset);
    }
    else{
      memory_v *srcHandle  = mHandle;
      memory_v *destHandle = dest.mHandle;

      const occa::mode modeS = srcHandle->mode();
      const occa::mode modeD = destHandle->mode();

      if(modeS & onChipMode) {
        destHandle->copyFrom(srcHandle->getMemoryHandle(),
                             bytes, srcOffset);
      }
      else if(modeD & onChipMode) {
        srcHandle->copyTo(destHandle->getMemoryHandle(),
                          bytes, destOffset);
      }
      else{
        OCCA_CHECK(((modeS == CUDA) && (modeD == CUDA)),
                   "Peer-to-peer is not supported between ["
                   << modeToStr(modeS) << "] and ["
                   << modeToStr(modeD) << "]");

#if OCCA_CUDA_ENABLED
        CUDADeviceData_t &srcDevData  =
          *((CUDADeviceData_t*) srcHandle->dHandle->data);

        CUDADeviceData_t &destDevData =
          *((CUDADeviceData_t*) destHandle->dHandle->data);

        CUdeviceptr srcMem  = *(((CUdeviceptr*) srcHandle->handle)  + srcOffset);
        CUdeviceptr destMem = *(((CUdeviceptr*) destHandle->handle) + destOffset);

        cuda::peerToPeerMemcpy(destDevData.device,
                               destDevData.context,
                               destMem,

                               srcDevData.device,
                               srcDevData.context,
                               srcMem,

                               bytes,
                               *((CUstream*) srcHandle->dHandle->currentStream));
#endif
      }
    }
  }

  void memory::asyncCopyFrom(const void *src,
                             const uintptr_t bytes,
                             const uintptr_t offset) {
    checkIfInitialized();
    mHandle->asyncCopyFrom(src, bytes, offset);
  }

  void memory::asyncCopyFrom(const memory src,
                             const uintptr_t bytes,
                             const uintptr_t destOffset,
                             const uintptr_t srcOffset) {
    checkIfInitialized();

    if(mHandle->dHandle == src.mHandle->dHandle) {
      mHandle->asyncCopyFrom(src.mHandle, bytes, destOffset, srcOffset);
    }
    else{
      memory_v *srcHandle  = src.mHandle;
      memory_v *destHandle = mHandle;

      const occa::mode modeS = srcHandle->mode();
      const occa::mode modeD = destHandle->mode();

      if(modeS & onChipMode) {
        destHandle->asyncCopyFrom(srcHandle->getMemoryHandle(),
                             bytes, destOffset);
      }
      else if(modeD & onChipMode) {
        srcHandle->asyncCopyTo(destHandle->getMemoryHandle(),
                          bytes, srcOffset);
      }
      else{
        OCCA_CHECK(((modeS == CUDA) && (modeD == CUDA)),
                   "Peer-to-peer is not supported between ["
                   << modeToStr(modeS) << "] and ["
                   << modeToStr(modeD) << "]");

#if OCCA_CUDA_ENABLED
        CUDADeviceData_t &srcDevData  =
          *((CUDADeviceData_t*) srcHandle->dHandle->data);

        CUDADeviceData_t &destDevData =
          *((CUDADeviceData_t*) destHandle->dHandle->data);

        CUdeviceptr srcMem  = *(((CUdeviceptr*) srcHandle->handle)  + srcOffset);
        CUdeviceptr destMem = *(((CUdeviceptr*) destHandle->handle) + destOffset);

        cuda::asyncPeerToPeerMemcpy(destDevData.device,
                                    destDevData.context,
                                    destMem,

                                    srcDevData.device,
                                    srcDevData.context,
                                    srcMem,

                                    bytes,
                                    *((CUstream*) srcHandle->dHandle->currentStream));
#endif
      }
    }
  }

  void memory::asyncCopyTo(void *dest,
                           const uintptr_t bytes,
                           const uintptr_t offset) {
    checkIfInitialized();
    mHandle->asyncCopyTo(dest, bytes, offset);
  }

  void memory::asyncCopyTo(memory dest,
                           const uintptr_t bytes,
                           const uintptr_t destOffset,
                           const uintptr_t srcOffset) {
    checkIfInitialized();

    if(mHandle->dHandle == dest.mHandle->dHandle) {
      mHandle->asyncCopyTo(dest.mHandle, bytes, destOffset, srcOffset);
    }
    else{
      memory_v *srcHandle  = mHandle;
      memory_v *destHandle = dest.mHandle;

      const occa::mode modeS = srcHandle->mode();
      const occa::mode modeD = destHandle->mode();

      if(modeS & onChipMode) {
        destHandle->asyncCopyFrom(srcHandle->getMemoryHandle(),
                                  bytes, destOffset);
      }
      else if(modeD & onChipMode) {
        srcHandle->asyncCopyTo(destHandle->getMemoryHandle(),
                               bytes, srcOffset);
      }
      else{
        OCCA_CHECK(((modeS == CUDA) && (modeD == CUDA)),
                   "Peer-to-peer is not supported between ["
                   << modeToStr(modeS) << "] and ["
                   << modeToStr(modeD) << "]");

#if OCCA_CUDA_ENABLED
        CUDADeviceData_t &srcDevData  =
          *((CUDADeviceData_t*) srcHandle->dHandle->data);

        CUDADeviceData_t &destDevData =
          *((CUDADeviceData_t*) destHandle->dHandle->data);

        CUdeviceptr srcMem  = *(((CUdeviceptr*) srcHandle->handle)  + srcOffset);
        CUdeviceptr destMem = *(((CUdeviceptr*) destHandle->handle) + destOffset);

        cuda::asyncPeerToPeerMemcpy(destDevData.device,
                                    destDevData.context,
                                    destMem,

                                    srcDevData.device,
                                    srcDevData.context,
                                    srcMem,

                                    bytes,
                                    *((CUstream*) srcHandle->dHandle->currentStream));
#endif
      }
    }
  }

  void memcpy(void *dest, void *src,
              const uintptr_t bytes,
              const int flags) {

    memcpy(dest, src, bytes, flags, false);
  }

  void asyncMemcpy(void *dest, void *src,
                   const uintptr_t bytes,
                   const int flags) {

    memcpy(dest, src, bytes, flags, true);
  }

  void memcpy(void *dest, void *src,
              const uintptr_t bytes,
              const int flags,
              const bool isAsync) {

    ptrRangeMap_t::iterator srcIt  = uvaMap.end();
    ptrRangeMap_t::iterator destIt = uvaMap.end();

    if(flags & occa::autoDetect) {
      srcIt  = uvaMap.find(src);
      destIt = uvaMap.find(dest);
    }
    else{
      if(flags & srcInUva)
        srcIt  = uvaMap.find(src);

      if(flags & destInUva)
        destIt  = uvaMap.find(dest);
    }

    occa::memory_v *srcMem  = ((srcIt != uvaMap.end())  ? (srcIt->second)  : NULL);
    occa::memory_v *destMem = ((destIt != uvaMap.end()) ? (destIt->second) : NULL);

    const uintptr_t srcOff  = (srcMem  ? (((char*) src)  - ((char*) srcMem->uvaPtr))  : 0);
    const uintptr_t destOff = (destMem ? (((char*) dest) - ((char*) destMem->uvaPtr)) : 0);

    const bool usingSrcPtr  = ((srcMem  == NULL) || srcMem->isManaged());
    const bool usingDestPtr = ((destMem == NULL) || destMem->isManaged());

    if(usingSrcPtr && usingDestPtr) {
      ::memcpy(dest, src, bytes);
    }
    else if(usingSrcPtr) {
      if(!isAsync)
        destMem->copyFrom(src, bytes, destOff);
      else
        destMem->asyncCopyFrom(src, bytes, destOff);
    }
    else if(usingDestPtr) {
      if(!isAsync)
        srcMem->copyTo(dest, bytes, srcOff);
      else
        srcMem->asyncCopyTo(dest, bytes, srcOff);
    }
    else {
      // Auto-detects peer-to-peer stuff
      occa::memory srcMemory(srcMem);
      occa::memory destMemory(destMem);

      if(!isAsync)
        srcMemory.copyTo(destMemory, bytes, destOff, srcOff);
      else
        srcMemory.asyncCopyTo(destMemory, bytes, destOff, srcOff);
    }
  }

  void memcpy(memory dest,
              const void *src,
              const uintptr_t bytes,
              const uintptr_t offset) {

    dest.copyFrom(src, bytes, offset);
  }

  void memcpy(void *dest,
              memory src,
              const uintptr_t bytes,
              const uintptr_t offset) {

    src.copyTo(dest, bytes, offset);
  }

  void memcpy(memory dest,
              memory src,
              const uintptr_t bytes,
              const uintptr_t destOffset,
              const uintptr_t srcOffset) {

    src.copyTo(dest, bytes, destOffset, srcOffset);
  }

  void asyncMemcpy(memory dest,
                   const void *src,
                   const uintptr_t bytes,
                   const uintptr_t offset) {

    dest.asyncCopyFrom(src, bytes, offset);
  }

  void asyncMemcpy(void *dest,
                   memory src,
                   const uintptr_t bytes,
                   const uintptr_t offset) {

    src.asyncCopyTo(dest, bytes, offset);
  }

  void asyncMemcpy(memory dest,
                   memory src,
                   const uintptr_t bytes,
                   const uintptr_t destOffset,
                   const uintptr_t srcOffset) {

    src.asyncCopyTo(dest, bytes, destOffset, srcOffset);
  }

  void memory::free() {
    checkIfInitialized();

    mHandle->dHandle->bytesAllocated -= (mHandle->size);

    if(mHandle->uvaPtr) {
      uvaMap.erase(mHandle->uvaPtr);
      mHandle->dHandle->uvaMap.erase(mHandle->uvaPtr);

      // CPU case where memory is shared
      if(mHandle->uvaPtr != mHandle->handle) {
        uvaMap.erase(mHandle->handle);
        mHandle->dHandle->uvaMap.erase(mHandle->uvaPtr);

        ::free(mHandle->uvaPtr);
        mHandle->uvaPtr = NULL;
      }
    }

    if(!mHandle->isMapped())
      mHandle->free();
    else
      mHandle->mappedFree();

    delete mHandle;
    mHandle = NULL;
  }
  //==============================================


  //---[ Device ]---------------------------------
  void stream::free() {
    if(dHandle == NULL)
      return;

    device(dHandle).freeStream(*this);
  }

  device::device() {
    dHandle = NULL;
  }

  device::device(device_v *dHandle_) :
    dHandle(dHandle_) {}

  device::device(deviceInfo &dInfo) {
    setup(dInfo);
  }

  device::device(const std::string &infos) {
    setup(infos);
  }

  device::device(const device &d) :
    dHandle(d.dHandle) {}

  device& device::operator = (const device &d) {
    dHandle = d.dHandle;

    return *this;
  }

  void* device::getContextHandle() {
    return dHandle->getContextHandle();
  }

  device_v* device::getDHandle() {
    return dHandle;
  }

  void device::setupHandle(occa::mode m) {
    switch(m) {

    case Serial:{
      dHandle = new device_t<Serial>();
      break;
    }
    case OpenMP:{
#if OCCA_OPENMP_ENABLED
      dHandle = new device_t<OpenMP>();
#else
      std::cout << "OCCA mode [OpenMP] is not enabled, defaulting to [Serial] mode\n";
      dHandle = new device_t<Serial>();
#endif
      break;
    }
    case OpenCL:{
#if OCCA_OPENCL_ENABLED
      dHandle = new device_t<OpenCL>();
#else
      std::cout << "OCCA mode [OpenCL] is not enabled, defaulting to [Serial] mode\n";
      dHandle = new device_t<Serial>();
#endif
      break;
    }
    case CUDA:{
#if OCCA_CUDA_ENABLED
      dHandle = new device_t<CUDA>();
#else
      std::cout << "OCCA mode [CUDA] is not enabled, defaulting to [Serial] mode\n";
      dHandle = new device_t<Serial>();
#endif
      break;
    }
    case Pthreads:{
      std::cout << "OCCA mode [Pthreads] is still in development-mode (unstable)\n";
      dHandle = new device_t<Pthreads>();
      break;
    }
    default:{
      std::cout << "Unsupported OCCA mode given, defaulting to [Serial] mode\n";
      dHandle = new device_t<Serial>();
    }
    }
  }

  void device::setupHandle(const std::string &m) {
    setupHandle( strToMode(m) );
  }

  void device::setup(deviceInfo &dInfo) {
    setup(dInfo.infos);
  }

  void device::setup(const std::string &infos) {
    argInfoMap aim(infos);

    OCCA_CHECK(aim.has("mode"),
               "OCCA mode not given");

    // Load [mode] from aim
    occa::mode m = strToMode(aim.get("mode"));

    setupHandle(m);

    dHandle->setup(aim);

    dHandle->modelID_ = library::deviceModelID(dHandle->getIdentifier());
    dHandle->id_      = library::genDeviceID();

    if(aim.has("UVA")) {
      if(upStringCheck(aim.get("UVA"), "enabled"))
        dHandle->uvaEnabled_ = true;
      else
        dHandle->uvaEnabled_ = false;
    }
    else
      dHandle->uvaEnabled_ = uvaEnabledByDefault_f;

    stream newStream = createStream();
    dHandle->currentStream = newStream.handle;
  }

  void device::setup(occa::mode m,
                     const int arg1, const int arg2) {
    setupHandle(m);

    argInfoMap aim;

    switch(m) {
    case Serial:{
      // Do Nothing
      break;
    }
    case OpenMP:{
      // Do Nothing, maybe add thread order next, dynamic static, etc
      break;
    }
    case OpenCL:{
      aim.set("platformID", arg1);
      aim.set("deviceID"  , arg2);
      break;
    }
    case CUDA:{
      aim.set("deviceID", arg1);
      break;
    }
    case Pthreads:{
      aim.set("threadCount", arg1);
      aim.set("pinningInfo", arg2);
      break;
    }
    }

    dHandle->setup(aim);

    dHandle->modelID_ = library::deviceModelID(dHandle->getIdentifier());
    dHandle->id_      = library::genDeviceID();

    stream newStream = createStream();
    dHandle->currentStream = newStream.handle;
  }


  void device::setup(const std::string &m,
                     const int arg1, const int arg2) {
    setup(strToMode(m), arg1, arg2);
  }

  uintptr_t device::memorySize() const {
    checkIfInitialized();
    return dHandle->memorySize();
  }

  uintptr_t device::memoryAllocated() const {
    checkIfInitialized();
    return dHandle->bytesAllocated;
  }

  // Old name for [memoryAllocated()]
  uintptr_t device::bytesAllocated() const {
    checkIfInitialized();
    return dHandle->bytesAllocated;
  }

  deviceIdentifier device::getIdentifier() const {
    checkIfInitialized();
    return dHandle->getIdentifier();
  }

  void device::setCompiler(const std::string &compiler_) {
    checkIfInitialized();
    dHandle->setCompiler(compiler_);
  }

  void device::setCompilerEnvScript(const std::string &compilerEnvScript_) {
    checkIfInitialized();
    dHandle->setCompilerEnvScript(compilerEnvScript_);
  }

  void device::setCompilerFlags(const std::string &compilerFlags_) {
    checkIfInitialized();
    dHandle->setCompilerFlags(compilerFlags_);
  }

  std::string& device::getCompiler() {
    checkIfInitialized();
    return dHandle->compiler;
  }

  std::string& device::getCompilerEnvScript() {
    checkIfInitialized();
    return dHandle->compilerEnvScript;
  }

  std::string& device::getCompilerFlags() {
    checkIfInitialized();
    return dHandle->compilerFlags;
  }

  int device::modelID() {
    checkIfInitialized();
    return dHandle->modelID_;
  }

  int device::id() {
    checkIfInitialized();
    return dHandle->id_;
  }

  int device::modeID() {
    checkIfInitialized();
    return dHandle->mode();
  }

  const std::string& device::mode() {
    checkIfInitialized();
    return dHandle->strMode;
  }

  void device::flush() {
    checkIfInitialized();
    dHandle->flush();
  }

  void device::finish() {
    checkIfInitialized();

    if(dHandle->fakesUva()) {
      const size_t dirtyEntries = uvaDirtyMemory.size();

      if(dirtyEntries) {
        for(size_t i = 0; i < dirtyEntries; ++i) {
          occa::memory_v *mem = uvaDirtyMemory[i];

          mem->asyncCopyTo(mem->uvaPtr);

          mem->memInfo &= ~uvaFlag::inDevice;
          mem->memInfo &= ~uvaFlag::isDirty;
        }

        uvaDirtyMemory.clear();
      }
    }

    dHandle->finish();
  }

  void device::waitFor(streamTag tag) {
    checkIfInitialized();
    dHandle->waitFor(tag);
  }

  stream device::createStream() {
    checkIfInitialized();

    stream newStream(dHandle, dHandle->createStream());

    dHandle->streams.push_back(newStream.handle);

    return newStream;
  }

  stream device::getStream() {
    checkIfInitialized();
    return stream(dHandle, dHandle->currentStream);
  }

  void device::setStream(stream s) {
    checkIfInitialized();
    dHandle->currentStream = s.handle;
  }

  stream device::wrapStream(void *handle_) {
    checkIfInitialized();
    return stream(dHandle, dHandle->wrapStream(handle_));
  }

  streamTag device::tagStream() {
    checkIfInitialized();
    return dHandle->tagStream();
  }

  double device::timeBetween(const streamTag &startTag, const streamTag &endTag) {
    checkIfInitialized();
    return dHandle->timeBetween(startTag, endTag);
  }

  void device::freeStream(stream s) {
    checkIfInitialized();

    const int streamCount = dHandle->streams.size();

    for(int i = 0; i < streamCount; ++i) {
      if(dHandle->streams[i] == s.handle) {
        if(dHandle->currentStream == s.handle)
          dHandle->currentStream = NULL;

        dHandle->freeStream(dHandle->streams[i]);
        dHandle->streams.erase(dHandle->streams.begin() + i);

        break;
      }
    }
  }

  kernel device::buildKernel(const std::string &str,
                             const std::string &functionName,
                             const kernelInfo &info_) {
    checkIfInitialized();

    if(sys::fileExists(str, flags::checkCacheDir))
      return buildKernelFromSource(str, functionName, info_);
    else
      return buildKernelFromString(str, functionName, info_);
  }

  kernel device::buildKernelFromString(const std::string &content,
                                       const std::string &functionName,
                                       const int language) {
    checkIfInitialized();

    return buildKernelFromString(content,
                                 functionName,
                                 defaultKernelInfo,
                                 language);
  }

  kernel device::buildKernelFromString(const std::string &content,
                                       const std::string &functionName,
                                       const kernelInfo &info_,
                                       const int language) {
    checkIfInitialized();

    kernelInfo info = info_;

    dHandle->addOccaHeadersToInfo(info);

    const std::string hash = getContentHash(content,
                                            dHandle->getInfoSalt(info));

    const std::string hashDir = hashDirFor("", hash);

    std::string stringSourceFile = hashDir;

    if(language & occa::usingOKL)
      stringSourceFile += "stringSource.okl";
    else if(language & occa::usingOFL)
      stringSourceFile += "stringSource.ofl";
    else
      stringSourceFile += "stringSource.occa";

    if(!haveHash(hash, 1)) {
      waitForHash(hash, 1);

      return buildKernelFromBinary(hashDir +
                                   dHandle->fixBinaryName(kc::binaryFile),
                                   functionName);
    }

    writeToFile(stringSourceFile, content);

    kernel k = buildKernelFromSource(stringSourceFile,
                                     functionName,
                                     info_);

    releaseHash(hash, 1);

    return k;
  }

  kernel device::buildKernelFromSource(const std::string &filename,
                                       const std::string &functionName,
                                       const kernelInfo &info_) {
    checkIfInitialized();

    const std::string realFilename = sys::getFilename(filename);
    const bool usingParser         = fileNeedsParser(filename);

    kernel ker;

    kernel_v *&k = ker.kHandle;

    if(usingParser) {
#if OCCA_OPENMP_ENABLED
      if(dHandle->mode() != OpenMP) {
        k          = new kernel_t<Serial>;
        k->dHandle = new device_t<Serial>;
      }
      else {
        k          = new kernel_t<OpenMP>;
        k->dHandle = dHandle;
      }
#else
      k          = new kernel_t<Serial>;
      k->dHandle = new device_t<Serial>;
#endif

      const std::string hash = getFileContentHash(realFilename,
                                                  dHandle->getInfoSalt(info_));

      const std::string hashDir    = hashDirFor(realFilename, hash);
      const std::string parsedFile = hashDir + "parsedSource.occa";

      k->metaInfo = parseFileForFunction(mode(),
                                         realFilename,
                                         parsedFile,
                                         functionName,
                                         info_);

      kernelInfo info = defaultKernelInfo;
      info.addDefine("OCCA_LAUNCH_KERNEL", 1);

      k->buildFromSource(parsedFile, functionName, info);
      k->nestedKernels.clear();

      if (k->metaInfo.nestedKernels) {
        std::stringstream ss;

        const int vc_f = verboseCompilation_f;

        for(int ki = 0; ki < k->metaInfo.nestedKernels; ++ki) {
          ss << ki;

          const std::string sKerName = k->metaInfo.baseName + ss.str();

          ss.str("");

          kernel sKer;
          sKer.kHandle = dHandle->buildKernelFromSource(parsedFile,
                                                        sKerName,
                                                        info_);

          sKer.kHandle->metaInfo               = k->metaInfo;
          sKer.kHandle->metaInfo.name          = sKerName;
          sKer.kHandle->metaInfo.nestedKernels = 0;
          sKer.kHandle->metaInfo.removeArg(0); // remove nestedKernels **
          k->nestedKernels.push_back(sKer);

          // Only show compilation the first time
          if(ki == 0)
            verboseCompilation_f = false;
        }

        verboseCompilation_f = vc_f;
      }
    }
    else{
      k = dHandle->buildKernelFromSource(realFilename,
                                         functionName,
                                         info_);
      k->dHandle = dHandle;
    }

    return ker;
  }

  kernel device::buildKernelFromBinary(const std::string &filename,
                                       const std::string &functionName) {
    checkIfInitialized();

    kernel ker;
    ker.kHandle = dHandle->buildKernelFromBinary(filename, functionName);
    ker.kHandle->dHandle = dHandle;

    return ker;
  }

  void device::cacheKernelInLibrary(const std::string &filename,
                                    const std::string &functionName,
                                    const kernelInfo &info_) {
    checkIfInitialized();
    dHandle->cacheKernelInLibrary(filename,
                                  functionName,
                                  info_);
  }

  kernel device::loadKernelFromLibrary(const char *cache,
                                       const std::string &functionName) {
    checkIfInitialized();

    kernel ker;
    ker.kHandle = dHandle->loadKernelFromLibrary(cache, functionName);
    ker.kHandle->dHandle = dHandle;

    return ker;
  }

  memory device::wrapMemory(void *handle_,
                            const uintptr_t bytes) {
    checkIfInitialized();

    memory mem;
    mem.mHandle = dHandle->wrapMemory(handle_, bytes);
    mem.mHandle->dHandle = dHandle;

    return mem;
  }

  void device::wrapManagedMemory(void *handle_,
                                 const uintptr_t bytes) {
    checkIfInitialized();
    memory mem = wrapMemory(handle_, bytes);
    mem.manage();
  }

  memory device::wrapTexture(void *handle_,
                             const int dim, const occa::dim &dims,
                             occa::formatType type, const int permissions) {
    checkIfInitialized();

    OCCA_CHECK((dim == 1) || (dim == 2),
               "Textures of [" << dim << "D] are not supported,"
               << "only 1D or 2D are supported at the moment");

    memory mem;
    mem.mHandle = dHandle->wrapTexture(handle_,
                                       dim, dims,
                                       type, permissions);
    mem.mHandle->dHandle = dHandle;

    return mem;
  }

  void device::wrapManagedTexture(void *handle_,
                                  const int dim, const occa::dim &dims,
                                  occa::formatType type, const int permissions) {
    checkIfInitialized();
    memory mem = wrapTexture(handle_, dim, dims, type, permissions);
    mem.manage();
  }

  memory device::malloc(const uintptr_t bytes,
                        void *src) {
    checkIfInitialized();

    memory mem;
    mem.mHandle          = dHandle->malloc(bytes, src);
    mem.mHandle->dHandle = dHandle;

    dHandle->bytesAllocated += bytes;

    return mem;
  }

  void* device::managedAlloc(const uintptr_t bytes,
                             void *src) {
    checkIfInitialized();

    memory mem = malloc(bytes, src);
    mem.manage();

    return mem.mHandle->uvaPtr;
  }

  memory device::textureAlloc(const int dim, const occa::dim &dims,
                              void *src,
                              occa::formatType type, const int permissions) {
    checkIfInitialized();

    OCCA_CHECK((dim == 1) || (dim == 2),
               "Textures of [" << dim << "D] are not supported,"
               << "only 1D or 2D are supported at the moment");

    OCCA_CHECK(src != NULL,
               "Non-NULL source is required for [textureAlloc] (texture allocation)");

    memory mem;

    mem.mHandle      = dHandle->textureAlloc(dim, dims, src, type, permissions);
    mem.mHandle->dHandle = dHandle;

    dHandle->bytesAllocated += (type.bytes() *
                                ((dim == 2) ?
                                 (dims[0] * dims[1]) :
                                 (dims[0]          )));

    return mem;
  }

  void* device::managedTextureAlloc(const int dim, const occa::dim &dims,
                                    void *src,
                                    occa::formatType type, const int permissions) {
    checkIfInitialized();

    memory mem = textureAlloc(dim, dims, src, type, permissions);

    mem.manage();

    return mem.mHandle->uvaPtr;
  }

  memory device::mappedAlloc(const uintptr_t bytes,
                             void *src) {
    checkIfInitialized();

    memory mem;

    mem.mHandle          = dHandle->mappedAlloc(bytes, src);
    mem.mHandle->dHandle = dHandle;

    dHandle->bytesAllocated += bytes;

    return mem;
  }

  void* device::managedMappedAlloc(const uintptr_t bytes,
                                   void *src) {
    checkIfInitialized();

    memory mem = mappedAlloc(bytes, src);

    mem.manage();

    return mem.mHandle->uvaPtr;
  }

  void device::free() {
    checkIfInitialized();

    const int streamCount = dHandle->streams.size();

    for(int i = 0; i < streamCount; ++i)
      dHandle->freeStream(dHandle->streams[i]);

    dHandle->free();

    delete dHandle;
    dHandle = NULL;
  }

  int device::simdWidth() {
    checkIfInitialized();

    return dHandle->simdWidth();
  }

  //   ---[ Device Functions ]----------
  device currentDevice;

  device getCurrentDevice() {
    if (currentDevice.getDHandle() == NULL) {
      currentDevice = host();
    }
    return currentDevice;
  }

  device host() {
    static device _host;
    if (_host.getDHandle() == NULL) {
      _host = device(new device_t<Serial>());
    }
    return _host;
  }

  void setDevice(device d) {
    currentDevice = d;
  }

  void setDevice(const std::string &infos) {
    currentDevice = device(infos);
  }

  mutex_t deviceListMutex;
  std::vector<device> deviceList;

  std::vector<device>& getDeviceList() {

    deviceListMutex.lock();

    if(deviceList.size()) {
      deviceListMutex.unlock();
      return deviceList;
    }

    device_t<Serial>::appendAvailableDevices(deviceList);

#if OCCA_OPENMP_ENABLED
    device_t<OpenMP>::appendAvailableDevices(deviceList);
#endif
#if OCCA_PTHREADS_ENABLED
    device_t<Pthreads>::appendAvailableDevices(deviceList);
#endif
#if OCCA_OPENCL_ENABLED
    device_t<OpenCL>::appendAvailableDevices(deviceList);
#endif
#if OCCA_CUDA_ENABLED
    device_t<CUDA>::appendAvailableDevices(deviceList);
#endif

    deviceListMutex.unlock();

    return deviceList;
  }

  void setCompiler(const std::string &compiler_) {
    currentDevice.setCompiler(compiler_);
  }

  void setCompilerEnvScript(const std::string &compilerEnvScript_) {
    currentDevice.setCompilerEnvScript(compilerEnvScript_);
  }

  void setCompilerFlags(const std::string &compilerFlags_) {
    currentDevice.setCompilerFlags(compilerFlags_);
  }

  std::string& getCompiler() {
    return currentDevice.getCompiler();
  }

  std::string& getCompilerEnvScript() {
    return currentDevice.getCompilerEnvScript();
  }

  std::string& getCompilerFlags() {
    return currentDevice.getCompilerFlags();
  }

  void flush() {
    currentDevice.flush();
  }

  void finish() {
    currentDevice.finish();
  }

  void waitFor(streamTag tag) {
    currentDevice.waitFor(tag);
  }

  stream createStream() {
    return currentDevice.createStream();
  }

  stream getStream() {
    return currentDevice.getStream();
  }

  void setStream(stream s) {
    currentDevice.setStream(s);
  }

  stream wrapStream(void *handle_) {
    return currentDevice.wrapStream(handle_);
  }

  streamTag tagStream() {
    return currentDevice.tagStream();
  }

  //   ---[ Kernel Functions ]----------

  kernel buildKernel(const std::string &str,
                     const std::string &functionName,
                     const kernelInfo &info_) {

    return currentDevice.buildKernel(str,
                                     functionName,
                                     info_);
  }

  kernel buildKernelFromString(const std::string &content,
                               const std::string &functionName,
                               const int language) {

    return currentDevice.buildKernelFromString(content,
                                               functionName,
                                               language);
  }

  kernel buildKernelFromString(const std::string &content,
                               const std::string &functionName,
                               const kernelInfo &info_,
                               const int language) {

    return currentDevice.buildKernelFromString(content,
                                               functionName,
                                               info_,
                                               language);
  }

  kernel buildKernelFromSource(const std::string &filename,
                               const std::string &functionName,
                               const kernelInfo &info_) {

    return currentDevice.buildKernelFromSource(filename,
                                               functionName,
                                               info_);
  }

  kernel buildKernelFromBinary(const std::string &filename,
                               const std::string &functionName) {

    return currentDevice.buildKernelFromBinary(filename,
                                               functionName);
  }

  void cacheKernelInLibrary(const std::string &filename,
                            const std::string &functionName,
                            const kernelInfo &info_) {

    return currentDevice.cacheKernelInLibrary(filename,
                                              functionName,
                                              info_);
  }

  kernel loadKernelFromLibrary(const char *cache,
                               const std::string &functionName) {

    return currentDevice.loadKernelFromLibrary(cache,
                                               functionName);
  }

  //   ---[ Memory Functions ]----------
  memory wrapMemory(void *handle_,
                    const uintptr_t bytes) {

    return currentDevice.wrapMemory(handle_, bytes);
  }

  void wrapManagedMemory(void *handle_,
                         const uintptr_t bytes) {

    currentDevice.wrapManagedMemory(handle_, bytes);
  }

  memory wrapTexture(void *handle_,
                     const int dim, const occa::dim &dims,
                     occa::formatType type, const int permissions) {

    return currentDevice.wrapTexture(handle_,
                                     dim, dims,
                                     type, permissions);
  }

  void wrapManagedTexture(void *handle_,
                          const int dim, const occa::dim &dims,
                          occa::formatType type, const int permissions) {

    currentDevice.wrapManagedTexture(handle_,
                                     dim, dims,
                                     type, permissions);
  }

  memory malloc(const uintptr_t bytes,
                void *src) {

    return currentDevice.malloc(bytes, src);
  }

  void* managedAlloc(const uintptr_t bytes,
                     void *src) {

    return currentDevice.managedAlloc(bytes, src);
  }

  memory textureAlloc(const int dim, const occa::dim &dims,
                      void *src,
                      occa::formatType type, const int permissions) {

    return currentDevice.textureAlloc(dim, dims,
                                      src,
                                      type, permissions);
  }

  void* managedTextureAlloc(const int dim, const occa::dim &dims,
                            void *src,
                            occa::formatType type, const int permissions) {

    return currentDevice.managedTextureAlloc(dim, dims,
                                             src,
                                             type, permissions);
  }

  memory mappedAlloc(const uintptr_t bytes,
                     void *src) {

    return currentDevice.mappedAlloc(bytes, src);
  }

  void* managedMappedAlloc(const uintptr_t bytes,
                           void *src) {

    return currentDevice.managedMappedAlloc(bytes, src);
  }
  //   =================================

  //   ---[ Free Functions ]------------
  void free(device d) {
    d.free();
  }

  void free(stream s) {
    currentDevice.freeStream(s);
  }

  void free(kernel k) {
    k.free();
  }

  void free(memory m) {
    m.free();
  }
  //   =================================

  void printAvailableDevices() {
    std::stringstream ss;
    ss << "==============o=======================o==========================================\n";
    ss << cpu::getDeviceListInfo();
#if OCCA_OPENCL_ENABLED
    ss << "==============o=======================o==========================================\n";
    ss << cl::getDeviceListInfo();
#endif
#if OCCA_CUDA_ENABLED
    ss << "==============o=======================o==========================================\n";
    ss << cuda::getDeviceListInfo();
#endif
    ss << "==============o=======================o==========================================\n";

    std::cout << ss.str();
  }

  deviceIdentifier::deviceIdentifier() :
    mode_(Serial) {}

  deviceIdentifier::deviceIdentifier(occa::mode m,
                                     const char *c, const size_t chars) {
    mode_ = m;
    load(c, chars);
  }

  deviceIdentifier::deviceIdentifier(occa::mode m, const std::string &s) {
    mode_ = m;
    load(s);
  }

  deviceIdentifier::deviceIdentifier(const deviceIdentifier &di) :
    mode_(di.mode_),
    flagMap(di.flagMap) {}

  deviceIdentifier& deviceIdentifier::operator = (const deviceIdentifier &di) {
    mode_ = di.mode_;
    flagMap = di.flagMap;

    return *this;
  }

  void deviceIdentifier::load(const char *c, const size_t chars) {
    const char *c1 = c;

    while((c1 < (c + chars)) && (*c1 != '\0')) {
      const char *c2 = c1;
      const char *c3;

      while(*c2 != '|')
        ++c2;

      c3 = (c2 + 1);

      while((c3 < (c + chars)) &&
            (*c3 != '\0') && (*c3 != '|'))
        ++c3;

      flagMap[std::string(c1, c2 - c1)] = std::string(c2 + 1, c3 - c2 - 1);

      c1 = (c3 + 1);
    }
  }

  void deviceIdentifier::load(const std::string &s) {
    return load(s.c_str(), s.size());
  }

  std::string deviceIdentifier::flattenFlagMap() const {
    std::string ret = "";

    cFlagMapIterator it = flagMap.begin();

    if(it == flagMap.end())
      return "";

    ret += it->first;
    ret += '|';
    ret += it->second;
    ++it;

    while(it != flagMap.end()) {
      ret += '|';
      ret += it->first;
      ret += '|';
      ret += it->second;

      ++it;
    }

    return ret;
  }

  int deviceIdentifier::compare(const deviceIdentifier &b) const {
    if(mode_ != b.mode_)
      return (mode_ < b.mode_) ? -1 : 1;

    cFlagMapIterator it1 =   flagMap.begin();
    cFlagMapIterator it2 = b.flagMap.begin();

    while(it1 != flagMap.end()) {
      const std::string &s1 = it1->second;
      const std::string &s2 = it2->second;

      const int cmp = s1.compare(s2);

      if(cmp)
        return cmp;

      ++it1;
      ++it2;
    }

    return 0;
  }
  //==============================================

  namespace cl {
    occa::device wrapDevice(void *platformIDPtr,
                            void *deviceIDPtr,
                            void *contextPtr) {
#if OCCA_OPENCL_ENABLED
      return cl::wrapDevice(*((cl_platform_id*) platformIDPtr),
                            *((cl_device_id*)   deviceIDPtr),
                            *((cl_context*)     contextPtr));
#else
      OCCA_CHECK(false,
                 "OCCA was not compiled with [OpenCL] enabled");

      return occa::host();
#endif
    }
  }

  namespace cuda {
    occa::device wrapDevice(void *devicePtr,
                            void *contextPtr) {
#if OCCA_CUDA_ENABLED
      return cuda::wrapDevice(*((CUdevice*) devicePtr),
                              *((CUcontext*) contextPtr));
#else
      OCCA_CHECK(false,
                 "OCCA was not compiled with [CUDA] enabled");

      return occa::host();
#endif
    }
  }
}
