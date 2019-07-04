#include <occa/core/device.hpp>
#include <occa/core/kernelBuilder.hpp>
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

  void kernelBuilder::run(occa::device device,
                          occa::scope scope) {
    occa::kernel &kernel = build(device, scope.props);
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
  strVector getInlinedKernelArgNames(const int argumentCount,
                                     const std::string &macroArgNames) {
    // Remove first and last () characters
    std::string source = strip(macroArgNames);
    source = source.substr(1, source.size() - 2);

    strVector names = (
      json::parse("[" + source + "]")
      .getArray<std::string>()
    );

    OCCA_ERROR("Incorrect argument count ["
               << names.size() << "] (Expected "
               << argumentCount << ")",
               argumentCount == (int) names.size());

    return names;
  }

  std::string formatInlinedArg(const inlinedKernel::arg_t &arg,
                               const std::string &argName) {
    std::stringstream ss;

    ss << arg.dtype << ' ';
    if (arg.isPointer) {
      ss << '*';
    }
    ss << argName;

    return ss.str();
  }

  std::string formatInlinedKernel(std::vector<inlinedKernel::arg_t> arguments,
                                  const std::string &macroArgNames,
                                  const std::string &macroKernel,
                                  const std::string &kernelName) {
    const int argumentCount = (int) arguments.size();

    // Remove first and last () characters
    std::string source = strip(macroKernel);
    source = source.substr(1, source.size() - 2);

    strVector argNames = getInlinedKernelArgNames(argumentCount, macroArgNames);

    std::stringstream ss;
    ss << "@kernel void " << kernelName << "(";
    for (int i = 0; i < argumentCount; ++i) {
      if (i) {
        ss << ", ";
      }
      ss << formatInlinedArg(arguments[i],
                             argNames[i]);
    }
    ss << ") {" << source << "}";

    return ss.str();
  }
  //====================================
}
