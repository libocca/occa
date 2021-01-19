#ifndef OCCA_FUNCTIONAL_BASEFUNCTION_HEADER
#define OCCA_FUNCTIONAL_BASEFUNCTION_HEADER

#include <occa/functional/functionDefinition.hpp>
#include <occa/functional/scope.hpp>
#include <occa/utils/hash.hpp>
#include <occa/types/typedefs.hpp>

namespace occa {
  class baseFunction {
  public:
    occa::scope scope;
    hash_t hash_;

    baseFunction(const occa::scope &scope_);

    functionDefinition& definition();

    virtual int argumentCount() const = 0;

    hash_t hash() const;

    operator hash_t () const;

    std::string buildFunctionCall(const std::string &functionName,
                                  const strVector &argumentValues) const;
  };
}

#endif
