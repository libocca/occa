#include <occa/core/device.hpp>
#include <occa/core/memory.hpp>
#include <occa/core/kernelArg.hpp>
#include <occa/tools/uva.hpp>

namespace occa {
  //---[ KernelArg ]--------------------
  kernelArgData::kernelArgData() :
    modeMemory(NULL),
    size(0),
    info(kArgInfo::none) {
    ::memset(&data, 0, sizeof(data));
  }

  kernelArgData::kernelArgData(const kernelArgData &k) {
    *this = k;
  }

  kernelArgData& kernelArgData::operator = (const kernelArgData &k) {
    modeMemory = k.modeMemory;

    data = k.data;
    size = k.size;
    info = k.info;

    return *this;
  }

  kernelArgData::~kernelArgData() {}

  occa::modeDevice_t* kernelArgData::getModeDevice() const {
    if (!modeMemory) {
      return NULL;
    }
    return modeMemory->modeDevice;
  }

  occa::modeMemory_t* kernelArgData::getModeMemory() const {
    return modeMemory;
  }

  void* kernelArgData::ptr() const {
    return ((info & kArgInfo::usePointer) ? data.void_ : (void*) &data);
  }

  void kernelArgData::setupForKernelCall(const bool isConst) const {
    if (!modeMemory              ||
        !modeMemory->isManaged() ||
        !modeMemory->modeDevice->hasSeparateMemorySpace()) {
      return;
    }
    if (!modeMemory->inDevice()) {
      modeMemory->copyFrom(modeMemory->uvaPtr, modeMemory->size);
      modeMemory->memInfo |= uvaFlag::inDevice;
    }
    if (!isConst && !modeMemory->isStale()) {
      uvaStaleMemory.push_back(modeMemory);
      modeMemory->memInfo |= uvaFlag::isStale;
    }
  }

  kernelArg::kernelArg() {}
  kernelArg::~kernelArg() {}

  kernelArg::kernelArg(const kernelArgData &arg) {
    args.push_back(arg);
  }

  kernelArg::kernelArg(const kernelArg &k) :
    args(k.args) {}

  int kernelArg::size() const {
    return (int) args.size();
  }

  const kernelArgData& kernelArg::operator [] (const int index) const {
    return args[index];
  }

  kernelArg& kernelArg::operator = (const kernelArg &k) {
    args = k.args;
    return *this;
  }

  kernelArg::kernelArg(const uint8_t arg) {
    kernelArgData kArg;
    kArg.data.uint8_ = arg;
    kArg.size        = sizeof(uint8_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const uint16_t arg) {
    kernelArgData kArg;
    kArg.data.uint16_ = arg;
    kArg.size         = sizeof(uint16_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const uint32_t arg) {
    kernelArgData kArg;
    kArg.data.uint32_ = arg;
    kArg.size         = sizeof(uint32_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const uint64_t arg) {
    kernelArgData kArg;
    kArg.data.uint64_ = arg;
    kArg.size         = sizeof(uint64_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const int8_t arg) {
    kernelArgData kArg;
    kArg.data.int8_ = arg;
    kArg.size       = sizeof(int8_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const int16_t arg) {
    kernelArgData kArg;
    kArg.data.int16_ = arg;
    kArg.size        = sizeof(int16_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const int32_t arg) {
    kernelArgData kArg;
    kArg.data.int32_ = arg;
    kArg.size        = sizeof(int32_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const int64_t arg) {
    kernelArgData kArg;
    kArg.data.int64_ = arg;
    kArg.size        = sizeof(int64_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const float arg) {
    kernelArgData kArg;
    kArg.data.float_ = arg;
    kArg.size         = sizeof(float);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const double arg) {
    kernelArgData kArg;
    kArg.data.double_ = arg;
    kArg.size         = sizeof(double);
    args.push_back(kArg);
  }

  void kernelArg::add(const kernelArg &arg) {
    const int newArgs = (int) arg.args.size();
    for (int i = 0; i < newArgs; ++i) {
      args.push_back(arg.args[i]);
    }
  }

  void kernelArg::add(void *arg,
                      bool lookAtUva, bool argIsUva) {
    add(arg, sizeof(void*), lookAtUva, argIsUva);
  }

  void kernelArg::add(void *arg, size_t bytes,
                      bool lookAtUva, bool argIsUva) {

    modeMemory_t *modeMemory = NULL;

    if (argIsUva) {
      modeMemory = (modeMemory_t*) arg;
    } else if (lookAtUva) {
      ptrRangeMap::iterator it = uvaMap.find(arg);
      if (it != uvaMap.end()) {
        modeMemory = it->second;
      }
    }

    if (modeMemory) {
      add(modeMemory->makeKernelArg());
    } else {
      kernelArgData kArg;
      kArg.data.void_ = arg;
      kArg.size       = bytes;
      kArg.info       = kArgInfo::usePointer;
      args.push_back(kArg);
    }
  }

  int kernelArg::argumentCount(const std::vector<kernelArg> &arguments) {
    const int kArgCount = (int) arguments.size();
    int argc = 0;
    for (int i = 0; i < kArgCount; ++i) {
      argc += arguments[i].args.size();
    }
    return argc;
  }
  //====================================
}
