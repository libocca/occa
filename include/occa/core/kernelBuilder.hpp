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
  template <class TM>
  dtype_t getMemoryDtype(const TM &arg) {
    if (typeMetadata<TM>::isPointer) {
      return dtype::get<TM>();
    }
    return dtype::none;
  }

  template <>
  dtype_t getMemoryDtype(const occa::memory &arg);

  template <class ARG1, class ARG2, class ARG3, class ARG4>
  std::vector<dtype_t> getInlinedKernelArgTypes(
    ARG1 arg1, ARG2 arg2, ARG3 arg3, ARG4 arg4
  ) {
    std::vector<dtype_t> types;
    types.reserve(8);
    types.push_back(dtype::get<ARG1>());
    types.push_back(getMemoryDtype(arg1));
    types.push_back(dtype::get<ARG2>());
    types.push_back(getMemoryDtype(arg2));
    types.push_back(dtype::get<ARG3>());
    types.push_back(getMemoryDtype(arg3));
    types.push_back(dtype::get<ARG4>());
    types.push_back(getMemoryDtype(arg4));
    return types;
  }

  std::string formatInlinedKernel(std::vector<dtype_t> arguments,
                                  const std::string &macroArgs,
                                  const std::string &macroKernel,
                                  const std::string &kernelName);
  //====================================
}

#endif
