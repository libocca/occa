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
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const uint16_t arg) {           \
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const uint32_t arg) {           \
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const uint64_t arg) {           \
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const int8_t arg) {             \
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const int16_t arg) {            \
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const int32_t arg) {            \
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const int64_t arg) {            \
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const float arg) {              \
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const double arg) {             \
    primitiveConstructor(arg);                      \
  }                                                 \
                                                    \
  inline CLASS_NAME(const std::nullptr_t arg) {     \
    pointerConstructor((void*) NULL, dtype::void_); \
  }                                                 \
                                                    \
  template <class T>                               \
  CLASS_NAME(T *arg) {                             \
    pointerConstructor(arg, dtype::get<T>());      \
  }                                                 \
                                                    \
  template <class T>                               \
  CLASS_NAME(const T *arg) {                       \
    pointerConstructor(arg, dtype::get<T>());      \
  }                                                 \
                                                    \
  void ignore_this_function_only_for_semicolon()
}

#endif
