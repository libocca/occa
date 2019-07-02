#include <occa/core/device.hpp>
#include <occa/core/kernelBuilder.hpp>
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
    occa::kernel &k = kernelMap[hash];
    if (!k.isInitialized()) {
      if (buildingFromFile) {
        k = device.buildKernel(source_, function_, props);
      } else {
        k = device.buildKernelFromString(source_, function_, props);
      }
    }
    return k;
  }

  occa::kernel kernelBuilder::operator [] (occa::device device) {
    return build(device,
                 hash(device));
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
                                     const std::string &macroArgs) {
    // Remove first and last () characters
    std::string source = strip(macroArgs);
    source = source.substr(1, source.size() - 2);

    strVector names;
    names.reserve(argumentCount);

    const char *cStart = source.c_str();
    const char *c = cStart;
    for (int i = 0; i < argumentCount; ++i) {
      lex::skipTo(c, ',');
      names.push_back(std::string(cStart, c - cStart));
      if (*c == '\0') {
        break;
      }
      cStart = ++c;
    }

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
                                  const std::string &macroArgs,
                                  const std::string &macroKernel,
                                  const std::string &kernelName) {
    const int argumentCount = (int) arguments.size();

    // Remove first and last () characters
    std::string source = strip(macroKernel);
    source = source.substr(1, source.size() - 2);

    strVector argNames = getInlinedKernelArgNames(argumentCount, macroArgs);

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
