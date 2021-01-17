#ifndef OCCA_EXPERIMENTAL_FUNCTIONAL_UTILS_HEADER
#define OCCA_EXPERIMENTAL_FUNCTIONAL_UTILS_HEADER

#define OCCA_FUNCTION(scope, lambda)             \
  ::occa::inferFunction(scope, lambda, #lambda)

namespace occa {
  template <class Function>
  class function;

  //---[ Magic ]------------------------
  // C++ template magic for casting between types at compile-time
  //   lambda
  //   -> std::function<ret(args...)>
  //   -> occa::function<ret(args...)>
  template <typename TM>
  struct inferFunctionHelper;

  template <typename ReturnType, typename ClassType, typename ...ArgTypes>
  struct inferFunctionHelper<ReturnType(ClassType::*)(ArgTypes...) const> {
    using occaFunctionType = occa::function<ReturnType(ArgTypes...)>;
  };

  template <typename LambdaType>
  typename inferFunctionHelper<decltype(&LambdaType::operator())>::occaFunctionType
  inferFunction(const occa::scope &scope,
                LambdaType const &lambda,
                const char *source) {
    return typename inferFunctionHelper<decltype(&LambdaType::operator())>::occaFunctionType(
      scope, lambda, source
    );
  }
  //====================================
}

#endif
