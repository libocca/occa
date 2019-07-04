#ifndef OCCA_CORE_SCOPE_HEADER
#define OCCA_CORE_SCOPE_HEADER

#include <map>

#include <occa/core/device.hpp>
#include <occa/core/kernelArg.hpp>
#include <occa/core/properties.hpp>

namespace occa {
  class scopeVariable;
  typedef std::vector<scopeVariable> scopeVariableVector;

  class scopeVariable {
   public:
    const std::string name;
    const kernelArg value;
    const bool isConst;

    scopeVariable(const std::string &name_,
                  const kernelArg &value_,
                  const bool isConst_);
  };

  class scope {
   public:
    occa::properties props;
    occa::device device;
    scopeVariableVector args;

    scope();
    scope(const occa::properties &props_);

    template <class TM>
    void addConst(const std::string &name,
                  const TM &value) {
      args.push_back(scopeVariable(name, value, true));
    }

    template <class TM>
    void add(const std::string &name,
             const TM &value) {
      args.push_back(scopeVariable(name, value, false));
    }

    occa::device getDevice();

    kernelArg getArg(const std::string &name);
  };
}

#endif
