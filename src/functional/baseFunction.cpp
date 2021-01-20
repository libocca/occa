#include <occa/functional/baseFunction.hpp>
#include <occa/internal/functional/functionStore.hpp>

namespace occa {
  baseFunction::baseFunction(const occa::scope &scope_) :
    scope(scope_) {}

  functionDefinition& baseFunction::definition() {
    // Should be initialized at this point
    return *functionStore.get(hash_);
  }

  const functionDefinition& baseFunction::definition() const {
    // Should be initialized at this point
    return *functionStore.get(hash_);
  }

  hash_t baseFunction::hash() const {
    return hash_;
  }

  baseFunction::operator hash_t () const {
    return hash_;
  }

  std::string baseFunction::buildFunctionCall(const std::string &functionName,
                                              const strVector &argumentValues) const {
    std::string call = functionName;
    call += '(';

    // Add the required arguments
    bool isFirst = true;
    for (const std::string &argumentValue : argumentValues) {
      if (!isFirst) {
        call += ", ";
      }
      call += argumentValue;
      isFirst = false;
    }

    // Add the scope-injected arguments
    for (const scopeKernelArg &arg : scope.args) {
      if (!isFirst) {
        call += ", ";
      }
      call += arg.name;
      isFirst = false;
    }

    call += ")";

    return call;
  }
}
