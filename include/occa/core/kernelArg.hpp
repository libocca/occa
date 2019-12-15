#ifndef OCCA_CORE_KERNELARG_HEADER
#define OCCA_CORE_KERNELARG_HEADER

#include <vector>

#include <occa/defines.hpp>
#include <occa/types.hpp>

namespace occa {
  class modeMemory_t; class memory;
  class modeDevice_t; class device;
  class kernelArgData;

  typedef std::vector<kernelArgData> kArgVector;

  //---[ KernelArg ]--------------------
  namespace kArgInfo {
    static const char none       = 0;
    static const char usePointer = (1 << 0);
    static const char isNull     = (1 << 1);
  }

  class nullKernelArg_t {
   public:
    inline nullKernelArg_t() {}
  };

  extern const nullKernelArg_t nullKernelArg;

  union kernelArgData_t {
    uint8_t  uint8_;
    uint16_t uint16_;
    uint32_t uint32_;
    uint64_t uint64_;

    int8_t  int8_;
    int16_t int16_;
    int32_t int32_;
    int64_t int64_;

    float float_;
    double double_;

    void* void_;
  };

  class kernelArgData {
  public:
    occa::modeMemory_t *modeMemory;

    kernelArgData_t data;
    udim_t size;
    char info;

    kernelArgData();
    kernelArgData(const kernelArgData &other);
    kernelArgData& operator = (const kernelArgData &other);
    ~kernelArgData();

    occa::modeDevice_t* getModeDevice() const;
    occa::modeMemory_t* getModeMemory() const;

    void* ptr() const;

    bool isNull() const;

    void setupForKernelCall(const bool isConst) const;
  };

  class kernelArg {
  public:
    kArgVector args;

    kernelArg();
    ~kernelArg();
    kernelArg(const kernelArgData &arg);
    kernelArg(const kernelArg &other);
    kernelArg& operator = (const kernelArg &other);

    kernelArg(const nullKernelArg_t arg);

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

    template <class TM>
    kernelArg(const type2<TM> &arg) {
      add((void*) const_cast<type2<TM>*>(&arg), sizeof(type2<TM>), false);
    }

    template <class TM>
    kernelArg(const type4<TM> &arg) {
      add((void*) const_cast<type4<TM>*>(&arg), sizeof(type4<TM>), false);
    }

    template <class TM>
    kernelArg(TM *arg) {
      add((void*) arg, true, false);
    }

    template <class TM>
    kernelArg(const TM *arg) {
      add((void*) const_cast<TM*>(arg), true, false);
    }

    int size() const;

    device getDevice() const;

    const kernelArgData& operator [] (const int index) const;

    void add(const kernelArg &arg);

    void add(void *arg,
             bool lookAtUva = true, bool argIsUva = false);

    void add(void *arg, size_t bytes,
             bool lookAtUva = true, bool argIsUva = false);

    static int argumentCount(const std::vector<kernelArg> &arguments);
  };

  template <>
  kernelArg::kernelArg(modeMemory_t *arg);

  template <>
  kernelArg::kernelArg(const modeMemory_t *arg);
  //====================================
}

#endif
