#ifndef OCCA_FUNCTIONAL_SCOPE_HEADER
#define OCCA_FUNCTIONAL_SCOPE_HEADER

#include <initializer_list>
#include <map>

#include <occa/core/device.hpp>
#include <occa/core/kernelArg.hpp>
#include <occa/types/primitive.hpp>
#include <occa/types/json.hpp>
#include <occa/utils/hash.hpp>

namespace occa {
  typedef std::initializer_list<scopeKernelArg> scopeKernelArgInitializerList;
  typedef std::vector<scopeKernelArg> scopeKernelArgVector;

  namespace scopeKernelArgMethods {
    template <class T>
    struct isMemory {
      static const bool value = false;
    };

    template <>
    struct isMemory<occa::memory> {
      static const bool value = true;
    };

    template <class T>
    struct argIsPointer {
      static const bool value = typeMetadata<T>::isPointer || isMemory<T>::value;
    };

    template <class T>
    inline dtype_t getPointerType(const T &arg) {
      if (typeMetadata<T>::isPointer) {
        return dtype::get<T>();
      }
      return dtype::none;
    }

    template <>
    inline dtype_t getPointerType<occa::memory>(const occa::memory &arg) {
      return arg.dtype();
    }
  }

  class scope {
  public:
    occa::json props;
    occa::device device;
    scopeKernelArgVector args;

    scope();

    scope(occa::device device_);

    scope(scopeKernelArgInitializerList args_,
          const occa::json &props_ = occa::json());

    scope(const occa::json &props_);

    inline void add(scopeKernelArg arg) {
      args.push_back(arg);

      occa::device argDevice = arg.getDevice();
      if (!argDevice.isInitialized()) {
        return;
      }

      if (!device.isInitialized()) {
        device = argDevice;
      } else if (device != argDevice) {
        OCCA_FORCE_ERROR("Device from arg [" << arg.name << "] doesn't match"
                         << " previous argument devices");
      }
    }

    template <class T>
    void add(const std::string &name,
             T *value) {
      add({name, value});
    }

    template <class T>
    void add(const std::string &name,
             const T *value) {
      // dtype -> name doesn't support const types yet
      add({name, value});
    }

    template <class T>
    void add(const std::string &name,
             T &value) {
      add({name, value});
    }

    template <class T>
    void add(const std::string &name,
             const T &value) {
      add({name, value});
    }

    scope operator + (const scope &other) const;
    scope& operator += (const scope &other);

    occa::device getDevice() const;

    int size() const;

    std::string getDeclarationSource() const;
    std::string getCallSource() const;

    kernelArg getArg(const std::string &name) const;

    hash_t hash() const;
  };

  template <>
  void scope::add(const std::string &name,
                  occa::memory &mem);

  template <>
  void scope::add(const std::string &name,
                  const occa::memory &mem);

  template <>
  hash_t hash(const occa::scope &scope);
}

#endif
