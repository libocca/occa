#include <occa/core/device.hpp>
#include <occa/types/json.hpp>
#include <occa/internal/core/kernel.hpp>
#include <occa/internal/lang/kernelMetadata.hpp>
#include <occa/internal/utils/lex.hpp>
#include <occa/internal/utils/string.hpp>
#include <occa/experimental/kernelBuilder.hpp>
#include <occa/functional/scope.hpp>

namespace occa {
  kernelBuilder::kernelBuilder(const std::string &source_,
                               const std::string &kernelName_) :
    source(strip(source_)),
    kernelName(strip(kernelName_)) {}

  bool kernelBuilder::isInitialized() {
    return (0 < kernelName.size());
  }

  std::string kernelBuilder::getKernelName() {
    return kernelName;
  }

  std::string kernelBuilder::buildKernelSource(const occa::scope &scope) {
    const int charCount = (int) source.size();

    // Remove first and last () characters
    if ((source[0] == '(') && (source[charCount - 1] == ')')) {
      source = source.substr(1, charCount - 2);
    }

    std::stringstream ss;
    ss << "@kernel void " << kernelName << "("
       << scope.getDeclarationSource()
       << ") {" << source << "}";

    return ss.str();
  }

  occa::kernel kernelBuilder::getOrBuildKernel(const occa::scope &scope) {
    occa::device device = scope.getDevice();
    const hash_t hash = (
      occa::hash(device) ^ occa::hash(scope)
    );

    occa::kernel &kernel = kernelMap[hash];
    if (!kernel.isInitialized()) {
      kernel = device.buildKernelFromString(
        buildKernelSource(scope),
        kernelName,
        scope.props
      );
    }
    return kernel;
  }

  void kernelBuilder::run(const occa::scope &scope) {
    occa::kernel kernel = getOrBuildKernel(scope);

    // Get argument metadata
    const lang::kernelMetadata_t &metadata = kernel.getModeKernel()->getMetadata();
    const std::vector<lang::argMetadata_t> &arguments = metadata.arguments;

    // Insert arguments in the proper order
    kernel.clearArgs();
    for (const lang::argMetadata_t &arg : arguments) {
      kernel.pushArg(scope.getArg(arg.name));
    }

    kernel.run();
  }

  void kernelBuilder::free() {
    hashedKernelMapIterator it = kernelMap.begin();
    while (it != kernelMap.end()) {
      it->second.free();
      ++it;
    }
    kernelMap.clear();
  }
}
