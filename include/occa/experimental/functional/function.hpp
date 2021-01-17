#ifndef OCCA_EXPERIMENTAL_FUNCTIONAL_FUNCTION_HEADER
#define OCCA_EXPERIMENTAL_FUNCTIONAL_FUNCTION_HEADER

#include <functional>

#include <occa/experimental/functional/functionDefinition.hpp>
#include <occa/experimental/functional/scope.hpp>
#include <occa/experimental/functional/utils.hpp>
#include <occa/utils/hash.hpp>
#include <occa/dtype.hpp>

namespace occa {
  template <class Function>
  class function;

  class baseFunction {
  public:
    occa::scope scope;
    hash_t hash_;

    baseFunction(const occa::scope &scope_);

    functionDefinition& definition();

    virtual int argumentCount() const = 0;

    hash_t hash() const;

    operator hash_t () const;
  };

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
