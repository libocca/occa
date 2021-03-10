#include <cstring>

#include <occa/core/device.hpp>
#include <occa/core/memory.hpp>
#include <occa/core/kernelArg.hpp>
#include <occa/utils/uva.hpp>
#include <occa/internal/core/memory.hpp>
#include <occa/internal/core/device.hpp>

namespace occa {
  //---[ kernelArgData ]----------------
  kernelArgData::kernelArgData() :
      value(),
      ptrSize(0),
      modeMemory(NULL) {}

  kernelArgData::kernelArgData(const primitive &value_) :
      value(value_),
      ptrSize(0),
      modeMemory(NULL) {}

  kernelArgData::kernelArgData(const kernelArgData &other) :
      value(other.value),
      ptrSize(other.ptrSize),
      modeMemory(other.modeMemory) {}

  kernelArgData& kernelArgData::operator = (const kernelArgData &other) {
    value = other.value;
    ptrSize = other.ptrSize;
    modeMemory = other.modeMemory;

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
    return const_cast<void*>(value.ptr());
  }

  udim_t kernelArgData::size() const {
    return ptrSize ? ptrSize : value.sizeof_();
  }

  bool kernelArgData::isPointer() const {
    return value.isPointer();
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
  //====================================

  //---[ kernelArg ]--------------------
  kernelArg::kernelArg() {}

  kernelArg::kernelArg(const kernelArgData &arg) {
    args.push_back(arg);
  }

  kernelArg::kernelArg(const kernelArg &other) :
      args(other.args) {}

  kernelArg& kernelArg::operator = (const kernelArg &other) {
    args = other.args;
    return *this;
  }

  kernelArg::~kernelArg() {}

  template <>
  kernelArg::kernelArg(modeMemory_t *arg) {
    addMemory(arg);
  }

  template <>
  kernelArg::kernelArg(const modeMemory_t *arg) {
    addMemory(const_cast<modeMemory_t*>(arg));
  }

  int kernelArg::size() const {
    return (int) args.size();
  }

  device kernelArg::getDevice() const {
    const int argCount = (int) args.size();

    for (int i = 0; i < argCount; ++i) {
      const kernelArgData &arg = args[i];
      if (arg.modeMemory) {
        return device(arg.modeMemory->modeDevice);
      }
    }

    return device();
  }

  const kernelArgData& kernelArg::operator [] (const int index) const {
    return args[index];
  }

  void kernelArg::add(const kernelArg &arg) {
    const int newArgs = (int) arg.args.size();
    for (int i = 0; i < newArgs; ++i) {
      args.push_back(arg.args[i]);
    }
  }

  void kernelArg::addPointer(void *arg,
                             bool lookAtUva, bool argIsUva) {
    addPointer(arg, sizeof(void*), lookAtUva, argIsUva);
  }

  void kernelArg::addPointer(void *arg, size_t bytes,
                             bool lookAtUva, bool argIsUva) {
    if (!arg) {
      args.push_back((primitive) nullptr);
      return;
    }

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
      addMemory(modeMemory);
    } else {
      kernelArgData kArg(arg);
      kArg.ptrSize = bytes;
      args.push_back(kArg);
    }
  }

  void kernelArg::addMemory(modeMemory_t *arg) {
    if (!arg || !arg->size) {
      add(nullptr);
      return;
    }

    // Set the modeMemory origin
    kernelArgData kArg(arg->getKernelArgPtr());
    kArg.modeMemory = arg;
    add(kArg);
  }

  int kernelArg::argumentCount(const std::vector<kernelArg> &arguments) {
    int argc = 0;
    for (auto &arg : arguments) {
      argc += arg.size();
    }
    return argc;
  }
  //====================================

  //---[ scopeKernelArg ]---------------
  scopeKernelArg::scopeKernelArg(const std::string &name_,
                                 const kernelArg &arg,
                                 const dtype_t &dtype_,
                                 const bool isConst_) :
    kernelArg(arg),
    name(name_),
    dtype(dtype_),
    isConst(isConst_) {}

  scopeKernelArg::scopeKernelArg(const std::string &name_,
                                 occa::memory &value_) :
    kernelArg(value_),
    name(name_),
    dtype(value_.dtype()),
    isConst(false) {}

  scopeKernelArg::scopeKernelArg(const std::string &name_,
                                 const occa::memory &value_) :
    kernelArg(value_),
    name(name_),
    dtype(value_.dtype()),
    isConst(false) {}

  scopeKernelArg::scopeKernelArg(const std::string &name_,
                                 const primitive &value_) :
    name(name_),
    isConst(true) {
    primitiveConstructor(value_);
  }

  scopeKernelArg::~scopeKernelArg() {}

  hash_t scopeKernelArg::hash() const {
    return occa::hash(getDeclaration());
  }

  std::string scopeKernelArg::getDeclaration() const {
    std::stringstream ss;

    bool isFirst = true;
    for (const kernelArgData &arg : args) {
      if (!isFirst) {
        ss << ", ";
      }

      if (isConst) {
        ss << "const ";
      }

      const dtype_t &safeDtype = dtype || dtype::void_;
      if (arg.isPointer()) {
        ss << safeDtype << " *";
      } else {
        ss << safeDtype << ' ';
      }

      ss << name;

      isFirst = false;
    }

    return ss.str();
  }

  template <>
  hash_t hash(const occa::scopeKernelArg &arg) {
    return arg.hash();
  }
  //====================================
}
