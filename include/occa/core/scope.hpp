#ifndef OCCA_CORE_SCOPE_HEADER
#define OCCA_CORE_SCOPE_HEADER

#include <map>

#include <occa/core/device.hpp>
#include <occa/core/kernelArg.hpp>
#include <occa/tools/properties.hpp>

namespace occa {
  class scopeVariable;
  typedef std::vector<scopeVariable> scopeVariableVector;

  namespace scopeVariableMethods {
    template <class TM>
    struct isMemory {
      static const bool value = false;
    };

    template <>
    struct isMemory<occa::memory> {
      static const bool value = true;
    };

    template <class TM>
    struct argIsPointer {
      static const bool value = typeMetadata<TM>::isPointer || isMemory<TM>::value;
    };

    template <class TM>
    inline dtype_t getPointerType(const TM &arg) {
      if (typeMetadata<TM>::isPointer) {
        return dtype::get<TM>();
      }
      return dtype::none;
    }

    template <>
    inline dtype_t getPointerType<occa::memory>(const occa::memory &arg) {
      return arg.dtype();
    }
  }

  class scopeVariable {
   public:
    dtype_t dtype;
    bool isPointer;
    bool isConst;
    std::string name;
    kernelArg value;

    inline scopeVariable() {}

    inline scopeVariable(const dtype_t &dtype_,
                         const bool &isPointer_,
                         const bool &isConst_,
                         const std::string &name_,
                         const kernelArg &value_) :
        dtype(dtype_),
        isPointer(isPointer_),
        isConst(isConst_),
        name(name_),
        value(value_) {}

    inline scopeVariable(const scopeVariable &other) :
        dtype(other.dtype),
        isPointer(other.isPointer),
        isConst(other.isConst),
        name(other.name),
        value(other.value) {}

    template <class TM>
    static scopeVariable fromValue(const std::string &name_,
                                   const TM &value_,
                                   const bool isConst_) {
      const bool isPointer_ = scopeVariableMethods::argIsPointer<TM>::value;
      dtype_t dtype_;
      if (isPointer_) {
        dtype_ = scopeVariableMethods::getPointerType<TM>(value_);
      } else {
        dtype_ = dtype::get<TM>();
      }

      return scopeVariable(dtype_, isPointer_, isConst_, name_, value_);
    }

    std::string getDeclaration() const;
  };

  class scope {
   public:
    occa::properties props;
    occa::device device;
    scopeVariableVector args;

    scope();
    scope(const occa::properties &props_);

    inline void add(scopeVariable arg) {
      args.push_back(arg);
      if (!device.isInitialized()) {
        device = arg.value.getDevice();
      }
    }

    template <class TM>
    void add(const std::string &name,
             TM *value) {
      add(scopeVariable::fromValue<TM*>(name, value, false));
    }

    template <class TM>
    void add(const std::string &name,
             const TM *value) {
      // dtype -> name doesn't support const types yet
      addConst(name, const_cast<TM*>(value));
    }

    template <class TM>
    void add(const std::string &name,
             TM &value) {
      add(scopeVariable::fromValue<TM>(name, value, false));
    }

    template <class TM>
    void add(const std::string &name,
             const TM &value) {
      addConst(name, value);
    }

    template <class TM>
    void addConst(const std::string &name,
                  const TM &value) {
      add(scopeVariable::fromValue<TM>(name, value, true));
    }

    occa::device getDevice();

    kernelArg getArg(const std::string &name);
  };
}

#endif
