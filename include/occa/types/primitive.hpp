#ifndef OCCA_TYPES_PRIMITIVE_HEADER
#define OCCA_TYPES_PRIMITIVE_HEADER

#include <iostream>
#include <sstream>
#include <iomanip>

#include <stdint.h>
#include <stdlib.h>

#include <occa/defines.hpp>
#include <occa/dtype/builtins.hpp>
#include <occa/utils/logging.hpp>

namespace occa {
  namespace io {
    class output;
  }

  //---[ Primitive Type ]---------------
  namespace primitiveType {
    static const int none       = (1 << 0);

    static const int bool_      = (1 << 1);

    static const int int8_      = (1 << 2);
    static const int uint8_     = (1 << 3);
    static const int int16_     = (1 << 4);
    static const int uint16_    = (1 << 5);
    static const int int32_     = (1 << 6);
    static const int uint32_    = (1 << 7);
    static const int int64_     = (1 << 8);
    static const int uint64_    = (1 << 9);

    static const int isSigned   = (int8_  |
                                   int16_ |
                                   int32_ |
                                   int64_);
    static const int isUnsigned = (uint8_  |
                                   uint16_ |
                                   uint32_ |
                                   uint64_);

    static const int isInteger  = (isSigned |
                                   isUnsigned);

    static const int float_     = (1 << 10);
    static const int double_    = (1 << 11);
    static const int isFloat    = (float_ |
                                   double_);

    static const int ptr        = (1 << 12);
  }
  //====================================

  class primitive {
  public:
    int type;
    std::string source;

    union {
      bool bool_;

      uint8_t  uint8_;
      uint16_t uint16_;
      uint32_t uint32_;
      uint64_t uint64_;

      int8_t  int8_;
      int16_t int16_;
      int32_t int32_;
      int64_t int64_;

      float  float_;
      double double_;

      char* ptr;
    } value;

    inline primitive() :
      type(primitiveType::none) {
      value.ptr = NULL;
    }

    inline primitive(const primitive &p) :
        type(p.type),
        source(p.source) {
      value.ptr = p.value.ptr;
    }

    inline primitive& operator = (const primitive &p) {
      type = p.type;
      source = p.source;
      value.ptr = p.value.ptr;
      return *this;
    }

    primitive(const char *c);
    primitive(const std::string &s);

    inline primitive(const bool value_) {
      type = primitiveType::bool_;
      value.bool_ = (bool) value_;
    }

    inline primitive(const uint8_t value_) {
      type = primitiveType::uint8_;
      value.uint8_ = (uint8_t) value_;
    }

    inline primitive(const uint16_t value_) {
      type = primitiveType::uint16_;
      value.uint16_ = (uint16_t) value_;
    }

    inline primitive(const uint32_t value_) {
      type = primitiveType::uint32_;
      value.uint32_ = (uint32_t) value_;
    }

    inline primitive(const uint64_t value_) {
      type = primitiveType::uint64_;
      value.uint64_ = (uint64_t) value_;
    }

    inline primitive(const int8_t value_) {
      type = primitiveType::int8_;
      value.int8_ = (int8_t) value_;
    }

    inline primitive(const int16_t value_) {
      type = primitiveType::int16_;
      value.int16_ = (int16_t) value_;
    }

    inline primitive(const int32_t value_) {
      type = primitiveType::int32_;
      value.int32_ = (int32_t) value_;
    }

    inline primitive(const int64_t value_) {
      type = primitiveType::int64_;
      value.int64_ = (int64_t) value_;
    }

    inline primitive(const float value_) {
      type = primitiveType::float_;
      value.float_ = value_;
    }

    inline primitive(const double value_) {
      type = primitiveType::double_;
      value.double_ = value_;
    }

    inline primitive(const void *value_) {
      type = primitiveType::ptr;
      value.ptr = (char*) const_cast<void*>(value_);
    }

    inline primitive(const std::nullptr_t &_) {
      type = primitiveType::ptr;
      value.ptr = NULL;
    }

    static primitive load(const char *&c,
                          const bool includeSign = true);
    static primitive load(const std::string &s,
                          const bool includeSign = true);

    static primitive loadBinary(const char *&c, const bool isNegative = false);
    static primitive loadHex(const char *&c, const bool isNegative = false);

    inline dtype_t dtype() const {
      switch(type) {
        case primitiveType::bool_   : return dtype::bool_;
        case primitiveType::uint8_  : return dtype::uint8;
        case primitiveType::uint16_ : return dtype::uint16;
        case primitiveType::uint32_ : return dtype::uint32;
        case primitiveType::uint64_ : return dtype::uint64;
        case primitiveType::int8_   : return dtype::int8;
        case primitiveType::int16_  : return dtype::int16;
        case primitiveType::int32_  : return dtype::int32;
        case primitiveType::int64_  : return dtype::int64;
        case primitiveType::float_  : return dtype::float_;
        case primitiveType::double_ : return dtype::double_;
        default: return dtype::none;
      }
    }

    inline primitive& operator = (const bool value_) {
      type = primitiveType::bool_;
      value.bool_ = (bool) value_;
      return *this;
    }

    inline primitive& operator = (const uint8_t value_) {
      type = primitiveType::uint8_;
      value.uint8_ = (uint8_t) value_;
      return *this;
    }

    inline primitive& operator = (const uint16_t value_) {
      type = primitiveType::uint16_;
      value.uint16_ = (uint16_t) value_;
      return *this;
    }

    inline primitive& operator = (const uint32_t value_) {
      type = primitiveType::uint32_;
      value.uint32_ = (uint32_t) value_;
      return *this;
    }

    inline primitive& operator = (const uint64_t value_) {
      type = primitiveType::uint64_;
      value.uint64_ = (uint64_t) value_;
      return *this;
    }

    inline primitive& operator = (const int8_t value_) {
      type = primitiveType::int8_;
      value.int8_ = (int8_t) value_;
      return *this;
    }

    inline primitive& operator = (const int16_t value_) {
      type = primitiveType::int16_;
      value.int16_ = (int16_t) value_;
      return *this;
    }

    inline primitive& operator = (const int32_t value_) {
      type = primitiveType::int32_;
      value.int32_ = (int32_t) value_;
      return *this;
    }

    inline primitive& operator = (const int64_t value_) {
      type = primitiveType::int64_;
      value.int64_ = (int64_t) value_;
      return *this;
    }

    inline primitive& operator = (const float value_) {
      type = primitiveType::float_;
      value.float_ = value_;
      return *this;
    }

    inline primitive& operator = (const double value_) {
      type = primitiveType::double_;
      value.double_ = value_;
      return *this;
    }

    inline primitive& operator = (void *value_) {
      type = primitiveType::ptr;
      value.ptr = (char*) value_;
      return *this;
    }

    inline operator bool () const {
      return to<bool>();
    }

    inline operator uint8_t () const {
      return to<uint8_t>();
    }

    inline operator uint16_t () const {
      return to<uint16_t>();
    }

    inline operator uint32_t () const {
      return to<uint32_t>();
    }

    inline operator uint64_t () const {
      return to<uint64_t>();
    }

    inline operator int8_t () const {
      return to<int8_t>();
    }

    inline operator int16_t () const {
      return to<int16_t>();
    }

    inline operator int32_t () const {
      return to<int32_t>();
    }

    inline operator int64_t () const {
      return to<int64_t>();
    }

    inline operator float () const {
      return to<float>();
    }

    inline operator double () const {
      return to<double>();
    }

    template <class T>
    inline T to() const {
      switch(type) {
      case primitiveType::bool_   : return (T) value.bool_;
      case primitiveType::uint8_  : return (T) value.uint8_;
      case primitiveType::uint16_ : return (T) value.uint16_;
      case primitiveType::uint32_ : return (T) value.uint32_;
      case primitiveType::uint64_ : return (T) value.uint64_;
      case primitiveType::int8_   : return (T) value.int8_;
      case primitiveType::int16_  : return (T) value.int16_;
      case primitiveType::int32_  : return (T) value.int32_;
      case primitiveType::int64_  : return (T) value.int64_;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
      case primitiveType::float_  : return (T) value.float_;
      case primitiveType::double_ : return (T) value.double_;
#pragma GCC diagnostic pop
      default: OCCA_FORCE_ERROR("Type not set");
      }
      return T();
    }

    inline bool isNaN() const {
      return type & primitiveType::none;
    }

    inline bool isBool() const {
      return type & primitiveType::bool_;
    }

    inline bool isSigned() const {
      return type & primitiveType::isSigned;
    }

    inline bool isUnsigned() const {
      return type & primitiveType::isUnsigned;
    }

    inline bool isInteger() const {
      return type & primitiveType::isInteger;
    }

    inline bool isFloat() const {
      return type & primitiveType::isFloat;
    }

    inline bool isPointer() const {
      return type & primitiveType::ptr;
    }

    inline bool isNull() const {
      return (isNaN() || (isPointer() && !value.ptr));
    }

    inline const void* ptr() const {
      switch (type) {
        case primitiveType::none: return NULL;
        case primitiveType::ptr: return value.ptr;
        default: return &value;
      }
    }

    inline void* ptr() {
      return const_cast<void*>(
        static_cast<const primitive*>(this)->ptr()
      );
    }

    std::string toString() const;
    std::string toBinaryString() const;
    std::string toHexString() const;

    friend io::output& operator << (io::output &out,
                                    const primitive &p);

    //---[ Misc Methods ]-----------------
    uint64_t sizeof_() const;
    //====================================

    //---[ Unary Operators ]--------------
    static primitive not_(const primitive &p);
    static primitive positive(const primitive &p);
    static primitive negative(const primitive &p);
    static primitive tilde(const primitive &p);
    static primitive& leftIncrement(primitive &p);
    static primitive& leftDecrement(primitive &p);
    static primitive rightIncrement(primitive &p);
    static primitive rightDecrement(primitive &p);
    //====================================


    //---[ Boolean Operators ]------------
    static primitive lessThan(const primitive &a, const primitive &b);
    static primitive lessThanEq(const primitive &a, const primitive &b);
    static primitive equal(const primitive &a, const primitive &b);
    static primitive compare(const primitive &a, const primitive &b);
    static primitive notEqual(const primitive &a, const primitive &b);
    static primitive greaterThanEq(const primitive &a, const primitive &b);
    static primitive greaterThan(const primitive &a, const primitive &b);
    static primitive and_(const primitive &a, const primitive &b);
    static primitive or_(const primitive &a, const primitive &b);
    //====================================


    //---[ Binary Operators ]-------------
    static primitive mult(const primitive &a, const primitive &b);
    static primitive add(const primitive &a, const primitive &b);
    static primitive sub(const primitive &a, const primitive &b);
    static primitive div(const primitive &a, const primitive &b);
    static primitive mod(const primitive &a, const primitive &b);
    static primitive bitAnd(const primitive &a, const primitive &b);
    static primitive bitOr(const primitive &a, const primitive &b);
    static primitive xor_(const primitive &a, const primitive &b);
    static primitive rightShift(const primitive &a, const primitive &b);
    static primitive leftShift(const primitive &a, const primitive &b);
    //====================================


    //---[ Assignment Operators ]---------
    static primitive& assign(primitive &a, const primitive &b);
    static primitive& multEq(primitive &a, const primitive &b);
    static primitive& addEq(primitive &a, const primitive &b);
    static primitive& subEq(primitive &a, const primitive &b);
    static primitive& divEq(primitive &a, const primitive &b);
    static primitive& modEq(primitive &a, const primitive &b);
    static primitive& bitAndEq(primitive &a, const primitive &b);
    static primitive& bitOrEq(primitive &a, const primitive &b);
    static primitive& xorEq(primitive &a, const primitive &b);
    static primitive& rightShiftEq(primitive &a, const primitive &b);
    static primitive& leftShiftEq(primitive &a, const primitive &b);
    //====================================
  };
}
#endif
