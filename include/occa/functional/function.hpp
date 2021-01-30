#ifndef OCCA_FUNCTIONAL_FUNCTION_HEADER
#define OCCA_FUNCTIONAL_FUNCTION_HEADER

#include <functional>

#include <occa/functional/baseFunction.hpp>
#include <occa/functional/utils.hpp>
#include <occa/dtype.hpp>

namespace occa {
  template <class Function>
  class function;

  /**
   * @startDoc{function}
   *
   * Description:
   *   Represents the input `std::function`, its metadata (such as types), and its source code.
   *
   *   ?> [[function]] objects should only be created through the `OCCA_FUNCTION` macro.
   *
   * @endDoc
   */
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

    /**
     * @startDoc{argumentCount}
     *
     * Description:
     *   Return how many arguments are expected to be passed
     *
     * @endDoc
     */
    int argumentCount() const {
      return (int) sizeof...(ArgTypes);
    }

    /**
     * @startDoc{getReturnType}
     *
     * Description:
     *   Return the [[dtype_t]] data type for the return value
     *
     * @endDoc
     */
    dtype_t getReturnType() const {
      return dtype::get<ReturnType>();
    }

    /**
     * @startDoc{getArgTypes}
     *
     * Description:
     *   Return a `std::vector`` of [[dtype_t]] data types, one for each argument expected
     *
     * @endDoc
     */
    dtypeVector getArgTypes() const {
      return dtype::getMany<ArgTypes...>();
    }

    /**
     * @startDoc{operator_parentheses}
     *
     * Description:
     *   Call the original `std::function`
     *
     * Argument Override:
     *    ArgTypes... args
     *
     * @endDoc
     */
    ReturnType operator () (ArgTypes... args) {
      return lambda(args...);
    }

    template <class T>
    friend class array;
  };
}

#endif
