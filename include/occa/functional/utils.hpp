#ifndef OCCA_FUNCTIONAL_UTILS_HEADER
#define OCCA_FUNCTIONAL_UTILS_HEADER

#include <occa/defines/macros.hpp>
#include <occa/functional/types.hpp>
#include <occa/functional/scope.hpp>

#define OCCA_FUNCTION_2(scope, lambda)                      \
  ::occa::functional::inferFunction(scope, lambda, #lambda)

#define OCCA_FUNCTION_1(lambda)                           \
  ::occa::functional::inferFunction({}, lambda, #lambda)

#define OCCA_FUNCTION(...)                                        \
  OCCA_FUNCTION_EXPAND_NAME_1(OCCA_ARG_COUNT(__VA_ARGS__)) (__VA_ARGS__)

#define OCCA_FUNCTION_EXPAND_NAME_1(ARG_COUNT)   \
  OCCA_FUNCTION_EXPAND_NAME_2(ARG_COUNT)

#define OCCA_FUNCTION_EXPAND_NAME_2(ARG_COUNT)   \
  OCCA_FUNCTION_EXPAND_NAME_3(ARG_COUNT)

#define OCCA_FUNCTION_EXPAND_NAME_3(ARG_COUNT) \
  OCCA_FUNCTION_ ## ARG_COUNT

namespace occa {
  template <class Function>
  class function;

  namespace functional {
    //---[ Magic ]------------------------
    // C++ template magic for casting between types at compile-time
    //   lambda
    //   -> std::function<ret(args...)>
    //   -> occa::function<ret(args...)>
    template <typename T>
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

    //---[ Array ]------------------------
    template <class T>
    T hostReduction(reductionType type, occa::memory mem) {
      const int entryCount = (int) mem.length();
      T *values = new T[entryCount];
      mem.copyTo(values);

      T reductionValue = values[0];
      switch (type) {
        case reductionType::sum:
          for (int i = 1; i < entryCount; ++i) {
            reductionValue += values[i];
          }
          break;
        case reductionType::multiply:
          for (int i = 1; i < entryCount; ++i) {
            reductionValue *= values[i];
          }
          break;
        case reductionType::bitOr:
          for (int i = 1; i < entryCount; ++i) {
            reductionValue |= values[i];
          }
          break;
        case reductionType::bitAnd:
          for (int i = 1; i < entryCount; ++i) {
            reductionValue &= values[i];
          }
          break;
        case reductionType::bitXor:
          for (int i = 1; i < entryCount; ++i) {
            reductionValue ^= values[i];
          }
          break;
        case reductionType::boolOr:
          for (int i = 1; i < entryCount; ++i) {
            reductionValue = reductionValue || values[i];
          }
          break;
        case reductionType::boolAnd:
          for (int i = 1; i < entryCount; ++i) {
            reductionValue = reductionValue && values[i];
          }
          break;
        case reductionType::min:
          for (int i = 1; i < entryCount; ++i) {
            reductionValue = reductionValue < values[i] ? reductionValue : values[i];
          }
          break;
        case reductionType::max:
          for (int i = 1; i < entryCount; ++i) {
            reductionValue = reductionValue > values[i] ? reductionValue : values[i];
          }
          break;
        default:
          break;
      }

      delete [] values;

      return reductionValue;
    }

    template <>
    bool hostReduction<bool>(reductionType type, occa::memory mem);

    template <>
    float hostReduction<float>(reductionType type, occa::memory mem);

    template <>
    double hostReduction<double>(reductionType type, occa::memory mem);
    //====================================
  }
}

#endif
