#ifndef OCCA_CORE_SCOPE_HEADER
#define OCCA_CORE_SCOPE_HEADER

#include <initializer_list>
#include <map>

#include <occa/core/device.hpp>
#include <occa/core/kernelArg.hpp>
#include <occa/types/primitive.hpp>
#include <occa/types/json.hpp>

namespace occa {
  typedef std::vector<scopeKernelArg> scopeKernelArgVector;

  namespace scopeKernelArgMethods {
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

  class scope {
   public:
    occa::json props;
    occa::device device;
    scopeKernelArgVector args;

    scope();

    scope(std::initializer_list<scopeKernelArg> args_,
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

    template <class TM>
    void add(const std::string &name,
             TM *value) {
      add({name, value});
    }

    template <class TM>
    void add(const std::string &name,
             const TM *value) {
      // dtype -> name doesn't support const types yet
      add({name, value});
    }

    template <class TM>
    void add(const std::string &name,
             TM &value) {
      add({name, value});
    }

    template <class TM>
    void add(const std::string &name,
             const TM &value) {
      add({name, value});
    }

    occa::device getDevice();

    kernelArg getArg(const std::string &name);
  };
}

#endif
