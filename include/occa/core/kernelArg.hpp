#ifndef OCCA_CORE_KERNELARG_HEADER
#define OCCA_CORE_KERNELARG_HEADER

#include <vector>

#include <occa/defines.hpp>
#include <occa/types.hpp>

namespace occa {
  class modeMemory_t; class memory;
  class modeDevice_t; class device;
  class kernelArgData;

  typedef std::vector<kernelArgData> kernelArgDataVector;

  //---[ KernelArg ]--------------------
  class kernelArgData {
  public:
    primitive value;
    udim_t ptrSize;
    occa::modeMemory_t *modeMemory;

    kernelArgData();
    kernelArgData(const primitive &value_);
    kernelArgData(const kernelArgData &other);
    kernelArgData& operator = (const kernelArgData &other);
    ~kernelArgData();

    occa::modeDevice_t* getModeDevice() const;
    occa::modeMemory_t* getModeMemory() const;

    udim_t size() const;
    void* ptr() const;

    void setupForKernelCall(const bool isConst) const;
  };

  class kernelArg {
  public:
    kernelArgDataVector args;

    kernelArg();
    ~kernelArg();
    kernelArg(const kernelArgData &arg);
    kernelArg(const kernelArg &other);
    kernelArg& operator = (const kernelArg &other);

    kernelArg(const uint8_t arg);
    kernelArg(const uint16_t arg);
    kernelArg(const uint32_t arg);
    kernelArg(const uint64_t arg);

    kernelArg(const int8_t arg);
    kernelArg(const int16_t arg);
    kernelArg(const int32_t arg);
    kernelArg(const int64_t arg);

    kernelArg(const float arg);
    kernelArg(const double arg);

    kernelArg(const std::nullptr_t arg);

    template <class TM>
    kernelArg(const type2<TM> &arg) {
      addPointer((void*) const_cast<type2<TM>*>(&arg), sizeof(type2<TM>), false);
    }

    template <class TM>
    kernelArg(const type4<TM> &arg) {
      addPointer((void*) const_cast<type4<TM>*>(&arg), sizeof(type4<TM>), false);
    }

    template <class TM>
    kernelArg(TM *arg) {
      addPointer((void*) arg, true, false);
    }

    template <class TM>
    kernelArg(const TM *arg) {
      addPointer((void*) const_cast<TM*>(arg), true, false);
    }

    int size() const;

    device getDevice() const;

    const kernelArgData& operator [] (const int index) const;

    void add(const kernelArg &arg);

    void addPointer(void *arg,
                    bool lookAtUva = true, bool argIsUva = false);

    void addPointer(void *arg, size_t bytes,
                    bool lookAtUva = true, bool argIsUva = false);

    void addMemory(modeMemory_t *arg);

    static int argumentCount(const std::vector<kernelArg> &arguments);
  };

  template <>
  kernelArg::kernelArg(modeMemory_t *arg);

  template <>
  kernelArg::kernelArg(const modeMemory_t *arg);
  //====================================
}

#endif
