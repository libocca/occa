#ifndef OCCA_EXPERIMENTAL_CORE_KERNELBUILDER_HEADER
#define OCCA_EXPERIMENTAL_CORE_KERNELBUILDER_HEADER

#include <occa/core/kernel.hpp>
#include <occa/experimental/scope.hpp>

namespace occa {
  class kernelBuilder {
  protected:
    std::string source_;
    std::string function_;
    occa::json defaultProps;

    hashedKernelMap kernelMap;

    bool buildingFromFile;

  public:
    kernelBuilder();

    kernelBuilder(const kernelBuilder &k);
    kernelBuilder& operator = (const kernelBuilder &k);

    const occa::json& defaultProperties() const;

    static kernelBuilder fromFile(const std::string &filename,
                                  const std::string &function,
                                  const occa::json &defaultProps_ = occa::json());

    static kernelBuilder fromString(const std::string &content,
                                    const std::string &function,
                                    const occa::json &defaultProps_ = occa::json());

    bool isInitialized();

    occa::kernel build(occa::device device);

    occa::kernel build(occa::device device,
                       const occa::json &props);

    occa::kernel build(occa::device device,
                       const hash_t &hash);

    occa::kernel build(occa::device device,
                       const hash_t &hash,
                       const occa::json &props);

    occa::kernel operator [] (occa::device device);

    void run(occa::scope &scope);

    void free();
  };
  //====================================


  //---[ Inlined Kernel ]---------------
  std::string formatInlinedKernelFromScope(occa::scope &scope,
                                           const std::string &oklSource,
                                           const std::string &kernelName);
  //====================================
}

#endif
