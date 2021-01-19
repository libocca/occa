#ifndef OCCA_FUNCTIONAL_FUNCTION_HEADER
#define OCCA_FUNCTIONAL_FUNCTION_HEADER

#include <functional>

#include <occa/functional/baseFunction.hpp>
#include <occa/functional/utils.hpp>
#include <occa/dtype.hpp>

namespace occa {
  template <class Function>
  class function;

  template <class ReturnType, class ...ArgTypes>
  class function<ReturnType(ArgTypes...)> : public baseFunction {
  private:
    std::function<ReturnType(ArgTypes...)> lambda;

  public:
    function(const occa::scope &scope_,
             std::function<ReturnType(ArgTypes...)> lambda_,
             const char *source) :
      baseFunction(scope_),
      lambda(lambda_) {

      hash_ = functionDefinition::cache(
        scope,
        source,
        getReturnType(),
        getArgTypes()
      ).get()->hash;
    }

    hash_t getTypeHash() const {
      hash_t typeHash = occa::hash(getReturnType().name());
      for (auto &argType : getArgTypes()) {
        typeHash ^= occa::hash(argType.name());
      }
      return typeHash;
    }

    int argumentCount() const {
      return (int) sizeof...(ArgTypes);
    }

    dtype_t getReturnType() const {
      return dtype::get<ReturnType>();
    }

    dtypeVector getArgTypes() const {
      return dtype::getMany<ArgTypes...>();
    }

    ReturnType operator () (ArgTypes... args) {
      return lambda(args...);
    }

    template <class TM>
    friend class array;
  };
}

#endif
