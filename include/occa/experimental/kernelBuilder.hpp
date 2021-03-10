#ifndef OCCA_EXPERIMENTAL_CORE_KERNELBUILDER_HEADER
#define OCCA_EXPERIMENTAL_CORE_KERNELBUILDER_HEADER

#include <occa/core/kernel.hpp>
#include <occa/functional/scope.hpp>

namespace occa {
  class kernelBuilder {
  private:
    std::string source;
    std::string kernelName;
    hashedKernelMap kernelMap;

  public:
    kernelBuilder(const std::string &source_,
                  const std::string &kernelName_);

    bool isInitialized();

    std::string getKernelName();

    std::string buildKernelSource(const occa::scope &scope);

    occa::kernel getOrBuildKernel(const occa::scope &scope);

    void run();
    void run(const occa::scope &scope);

    void free();
  };
}

#endif
