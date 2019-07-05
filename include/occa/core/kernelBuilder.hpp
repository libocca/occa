#ifndef OCCA_CORE_KERNELBUILDER_HEADER
#define OCCA_CORE_KERNELBUILDER_HEADER

#include <occa/core/kernel.hpp>

namespace occa {
  class scope;

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

    void run(occa::scope &scope);

    void free();
  };
  //====================================


  //---[ Inlined Kernel ]---------------
  std::string formatInlinedKernel(occa::scope &scope,
                                  const std::string &oklSource,
                                  const std::string &kernelName);
  //====================================
}

#endif
