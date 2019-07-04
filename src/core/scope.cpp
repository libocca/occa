#include <occa/core/scope.hpp>

namespace occa {
  scopeVariable::scopeVariable(const std::string &name_,
                               const kernelArg &value_,
                               const bool isConst_) :
      name(name_),
      value(value_),
      isConst(isConst_) {}

  scope::scope() {}

  scope::scope(const occa::properties &props_) :
      props(props_) {}

  occa::device scope::getDevice() {
    return device;
  }

  kernelArg scope::getArg(const std::string &name) {
    const int argCount = (int) args.size();

    for (int i = 0; i < argCount; ++i) {
      scopeVariable &arg = args[i];
      if (arg.name == name) {
        return arg.value;
      }
    }

    OCCA_FORCE_ERROR("Missing argument [" << name << "]");
  }
}

#endif
