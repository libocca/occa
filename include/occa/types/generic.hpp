#ifndef OCCA_TYPES_GENERIC_HEADER
#define OCCA_TYPES_GENERIC_HEADER

#include <occa/types/primitive.hpp>
#include <occa/dtype/builtins.hpp>

namespace occa {
  class generic {
   public:
    inline virtual ~generic() {}

   protected:
    virtual void primitiveConstructor(const primitive &value) = 0;
    virtual void pointerConstructor(void *ptr, const dtype_t &dtype_) = 0;
    virtual void pointerConstructor(const void *ptr, const dtype_t &dtype_) = 0;
  };

#define OCCA_GENERIC_CLASS_CONSTRUCTORS(CLASS_NAME) \
  inline CLASS_NAME(const uint8_t arg) {            \
    std::cout << "Making [uint8_t]\n";              \
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const uint16_t arg) {           \
    std::cout << "Making [uint16_t]\n";             \
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const uint32_t arg) {           \
    std::cout << "Making [uint32_t]\n";             \
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const uint64_t arg) {           \
    std::cout << "Making [uint64_t]\n";             \
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const int8_t arg) {             \
    std::cout << "Making [int8_t]\n";                 \
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const int16_t arg) {            \
    std::cout << "Making [int16_t]\n";              \
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const int32_t arg) {            \
    std::cout << "Making [int32_t]\n";              \
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const int64_t arg) {            \
    std::cout << "Making [int64_t]\n";              \
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const float arg) {              \
    std::cout << "Making [float]\n";                \
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const double arg) {             \
    std::cout << "Making [double]\n";               \
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const std::nullptr_t arg) {     \
    pointerConstructor((void*) NULL, dtype::void_); \
  }                                                 \
                                                    \
  template <class TM>                               \
  CLASS_NAME(TM *arg) {                             \
    pointerConstructor(arg, dtype::get<TM>());      \
  }                                                 \
                                                    \
  template <class TM>                               \
  CLASS_NAME(const TM *arg) {                       \
    pointerConstructor(arg, dtype::get<TM>());      \
  }                                                 \
                                                    \
  void ignore_this_function_only_for_semicolon()
}

#endif
