#ifndef OCCA_CORE_KERNELBUILDER_HEADER
#define OCCA_CORE_KERNELBUILDER_HEADER

#include <occa/core/kernel.hpp>

namespace occa {
  class kernelBuilder {
  protected:
    std::string source_;
    std::string function_;
    occa::properties props_;

    hashedKernelMap kernelMap;

    bool buildingFromFile;

  public:
    kernelBuilder();

    kernelBuilder(const kernelBuilder &k);
    kernelBuilder& operator = (const kernelBuilder &k);

    static kernelBuilder fromFile(const std::string &filename,
                                  const std::string &function,
                                  const occa::properties &props = occa::properties());

    static kernelBuilder fromString(const std::string &content,
                                    const std::string &function,
                                    const occa::properties &props = occa::properties());

    bool isInitialized();

    occa::kernel build(occa::device device);

    occa::kernel build(occa::device device,
                       const occa::properties &props);

    occa::kernel build(occa::device device,
                       const hash_t &hash);

    occa::kernel build(occa::device device,
                       const hash_t &hash,
                       const occa::properties &props);

    occa::kernel operator [] (occa::device device);

    void free();
  };
  //====================================


  //---[ Inlined Kernel ]---------------
  namespace inlinedKernel {
    class arg_t {
     public:
      dtype_t dtype;
      bool isPointer;

      inline arg_t(const dtype_t &dtype_,
                   const bool isPointer_) :
          dtype(dtype_),
          isPointer(isPointer_) {}
    };

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

    template <class TM>
    void addArg(std::vector<arg_t> &types,
                TM arg) {
      if (argIsPointer<TM>::value) {
        types.push_back(
          arg_t(getPointerType<TM>(arg), true)
        );
      } else {
        types.push_back(
          arg_t(dtype::get<TM>(), false)
        );
      }
    }
  }

#include "inlinedKernelArgTypes.hpp"

  std::string formatInlinedKernel(std::vector<inlinedKernel::arg_t> arguments,
                                  const std::string &macroArgNames,
                                  const std::string &macroKernel,
                                  const std::string &kernelName);
  //====================================
}

#endif
