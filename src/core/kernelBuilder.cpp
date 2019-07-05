#include <occa/core/device.hpp>
#include <occa/core/kernelBuilder.hpp>
#include <occa/core/scope.hpp>
#include <occa/tools/json.hpp>
#include <occa/tools/lex.hpp>
#include <occa/tools/string.hpp>

namespace occa {
  //---[ kernelBuilder ]----------------
  kernelBuilder::kernelBuilder() {}

  kernelBuilder::kernelBuilder(const kernelBuilder &k) :
    source_(k.source_),
    function_(k.function_),
    props_(k.props_),
    kernelMap(k.kernelMap),
    buildingFromFile(k.buildingFromFile) {}

  kernelBuilder& kernelBuilder::operator = (const kernelBuilder &k) {
    source_   = k.source_;
    function_ = k.function_;
    props_    = k.props_;
    kernelMap = k.kernelMap;
    buildingFromFile = k.buildingFromFile;
    return *this;
  }

  kernelBuilder kernelBuilder::fromFile(const std::string &filename,
                                        const std::string &function,
                                        const occa::properties &props) {
    kernelBuilder builder;
    builder.source_   = filename;
    builder.function_ = function;
    builder.props_    = props;
    builder.buildingFromFile = true;
    return builder;
  }

  kernelBuilder kernelBuilder::fromString(const std::string &content,
                                          const std::string &function,
                                          const occa::properties &props) {
    kernelBuilder builder;
    builder.source_   = content;
    builder.function_ = function;
    builder.props_    = props;
    builder.buildingFromFile = false;
    return builder;
  }

  bool kernelBuilder::isInitialized() {
    return (0 < function_.size());
  }

  occa::kernel kernelBuilder::build(occa::device device) {
    return build(device, hash(device), props_);
  }

  occa::kernel kernelBuilder::build(occa::device device,
                                    const occa::properties &props) {
    occa::properties kernelProps = props_;
    kernelProps += props;
    return build(device,
                 hash(device) ^ hash(kernelProps),
                 kernelProps);
  }

  occa::kernel kernelBuilder::build(occa::device device,
                                    const hash_t &hash) {
    return build(device, hash, props_);
  }

  occa::kernel kernelBuilder::build(occa::device device,
                                    const hash_t &hash,
                                    const occa::properties &props) {
    occa::kernel &kernel = kernelMap[hash];
    if (!kernel.isInitialized()) {
      if (buildingFromFile) {
        kernel = device.buildKernel(source_, function_, props);
      } else {
        kernel = device.buildKernelFromString(source_, function_, props);
      }
    }
    return kernel;
  }

  occa::kernel kernelBuilder::operator [] (occa::device device) {
    return build(device, hash(device));
  }

  void kernelBuilder::run(occa::scope &scope) {
    occa::kernel kernel = build(scope.getDevice(),
                                scope.props);
    kernel.clearArgs();

    // Get argument metadata
    lang::kernelMetadata &metadata = kernel.getModeKernel()->metadata;
    std::vector<lang::argumentInfo> arguments = metadata.arguments;

    // Insert arguments in the proper order
    const int argCount = (int) arguments.size();
    for (int i = 0; i < argCount; ++i) {
      lang::argumentInfo &arg = arguments[i];
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
  }
  //====================================


  //---[ Inlined Kernel ]---------------
  std::string formatInlinedKernel(occa::scope &scope,
                                  const std::string &oklSource,
                                  const std::string &kernelName) {
    std::string source = strip(oklSource);
    const int charCount = (int) source.size();

    // Remove first and last () characters
    if ((source[0] == '(') && (source[charCount - 1] == ')')) {
      source = source.substr(1, charCount - 2);
    }

    scopeVariableVector &args = scope.args;
    const int argCount = (int) args.size();

    std::stringstream ss;
    ss << "@kernel void " << kernelName << "(";
    for (int i = 0; i < argCount; ++i) {
      if (i) {
        ss << ", ";
      }
      ss << args[i].getDeclaration();
    }
    ss << ") {" << source << "}";

    return ss.str();
  }
  //====================================
}
