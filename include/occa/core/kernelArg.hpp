#ifndef OCCA_CORE_KERNELARG_HEADER
#define OCCA_CORE_KERNELARG_HEADER

#include <vector>

#include <occa/defines.hpp>
#include <occa/types/generic.hpp>
#include <occa/types/primitive.hpp>
#include <occa/utils/hash.hpp>
#include <occa/dtype.hpp>

namespace occa {
  class modeMemory_t; class memory;
  class modeDevice_t; class device;
  class kernelArgData;

  typedef std::vector<kernelArgData> kernelArgDataVector;

  //---[ kernelArgData ]----------------
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

    bool isPointer() const;

    void setupForKernelCall(const bool isConst) const;
  };
  //====================================

  //---[ kernelArg ]--------------------
  /**
   * @startDoc{kernelArg}
   *
   * Description:
   *   [[kernel]] arguments must be of type [[kernelArg]].
   *   Custom user types can be passed to a kernel by implementing a cast operator such as:
   *
   *   ```cpp
   *   operator occa::kernelArg() const;
   *   ```
   *
   * @endDoc
   */
  class kernelArg : public generic {
   public:
    kernelArgDataVector args;

    kernelArg();
    kernelArg(const kernelArgData &arg);
    kernelArg(const kernelArg &other);
    kernelArg& operator = (const kernelArg &other);

    virtual ~kernelArg();

    inline virtual void primitiveConstructor(const primitive &value) {
      args.push_back(value);
    }

    inline virtual void pointerConstructor(void *ptr, const dtype_t &dtype_) {
      addPointer(ptr, sizeof(void*), true, false);
    }

    inline virtual void pointerConstructor(const void *ptr, const dtype_t &dtype_) {
      addPointer(const_cast<void*>(ptr), sizeof(void*), true, false);
    }

    OCCA_GENERIC_CLASS_CONSTRUCTORS(kernelArg);

    template <class T>
    kernelArg(const type2<T> &arg) {
      addPointer((void*) const_cast<type2<T>*>(&arg), sizeof(type2<T>), false);
    }

    template <class T>
    kernelArg(const type4<T> &arg) {
      addPointer((void*) const_cast<type4<T>*>(&arg), sizeof(type4<T>), false);
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

  //---[ scopeKernelArg ]---------------
  class scopeKernelArg : public kernelArg {
   public:
    std::string name;
    dtype_t dtype;
    bool isConst;

    scopeKernelArg(const std::string &name_,
                   const kernelArg &arg,
                   const dtype_t &dtype_,
                   const bool isConst_);

    scopeKernelArg(const std::string &name_,
                   occa::memory &value_);

    scopeKernelArg(const std::string &name_,
                   const occa::memory &value_);

    scopeKernelArg(const std::string &name_,
                   const primitive &value_);

    template <class T>
    inline scopeKernelArg(const std::string &name_,
                          T *value_) :
      name(name_),
      isConst(false) {
      pointerConstructor(value_, dtype::get<T>());
    }

    template <class T>
    inline scopeKernelArg(const std::string &name_,
                          const T *value_) :
      name(name_),
      isConst(true) {
      pointerConstructor(value_, dtype::get<T>());
    }

    virtual ~scopeKernelArg();

    hash_t hash() const;

    inline void primitiveConstructor(const primitive &value) {
      dtype = value.dtype();
      isConst = true;

      kernelArg::primitiveConstructor(value);
    }

    inline void pointerConstructor(void *ptr, const dtype_t &dtype_) {
      dtype = dtype_;
      isConst = false;

      kernelArg::pointerConstructor(ptr, dtype_);
    }

    inline void pointerConstructor(const void *ptr, const dtype_t &dtype_) {
      dtype = dtype_;
      isConst = true;

      kernelArg::pointerConstructor(ptr, dtype_);
    }

    OCCA_GENERIC_CLASS_CONSTRUCTORS(scopeKernelArg);

    std::string getDeclaration() const;
  };

  template <>
  hash_t hash(const occa::scopeKernelArg &arg);
  //====================================
}

#endif
