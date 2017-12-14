/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#ifndef OCCA_PARSER_PRIMITIVE_HEADER
#define OCCA_PARSER_PRIMITIVE_HEADER

#include <iostream>
#include <sstream>
#include <iomanip>

#include <stdint.h>
#include <stdlib.h>

#include "occa/defines.hpp"
#include "occa/tools/string.hpp"
#include "occa/tools/sys.hpp"

namespace occa {
  class primitive {
  public:
    enum type_t {
      none        = (1 << 0),

      isSigned_   = (0x55 << 1),
      isUnsigned_ = (0x55 << 2),
      isInteger_  = (0xFF << 1),

      int8_       = (1 << 1),
      uint8_      = (1 << 2),
      int16_      = (1 << 3),
      uint16_     = (1 << 4),
      int32_      = (1 << 5),
      uint32_     = (1 << 6),
      int64_      = (1 << 7),
      uint64_     = (1 << 8),

      isFloat     = (3 << 9),
      float_      = (1 << 9),
      double_     = (1 << 10),

      ptr         = (1 << 11)
    };

    type_t type;

    union {
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
      type(none) {
      value.ptr = NULL;
    }

    inline primitive(const primitive &p) :
      type(p.type) {
      value.ptr = p.value.ptr;
    }

    primitive(const char *c);
    primitive(const std::string &s);

    inline primitive(const uint8_t value_) {
      type = uint8_;
      value.uint8_ = (uint8_t) value_;
    }

    inline primitive(const uint16_t value_) {
      type = uint16_;
      value.uint16_ = (uint16_t) value_;
    }

    inline primitive(const uint32_t value_) {
      type = uint32_;
      value.uint32_ = (uint32_t) value_;
    }

    inline primitive(const uint64_t value_) {
      type = uint64_;
      value.uint64_ = (uint64_t) value_;
    }

    inline primitive(const int8_t value_) {
      type = int8_;
      value.int8_ = (int8_t) value_;
    }

    inline primitive(const int16_t value_) {
      type = int16_;
      value.int16_ = (int16_t) value_;
    }

    inline primitive(const int32_t value_) {
      type = int32_;
      value.int32_ = (int32_t) value_;
    }

    inline primitive(const int64_t value_) {
      type = int64_;
      value.int64_ = (int64_t) value_;
    }

    inline primitive(const float value_) {
      type = float_;
      value.float_ = value_;
    }

    inline primitive(const double value_) {
      type = double_;
      value.double_ = value_;
    }

    inline primitive(void *value_) {
      type = ptr;
      value.ptr = (char*) value_;
    }

    static inline bool charIsDelimiter(const char c) {
      static const char delimiters[]  = " \t\r\n\v\f!\"#%&'()*+,-./:;<=>?[]^{|}~@\0";
      const char *d = delimiters;
      while (*d != '\0')
        if (c == *(d++)) {
          return true;
        }
      return false;
    }

    static primitive load(const char *&c);
    static primitive load(const std::string &s);

    static primitive loadBinary(const char *&c, const bool isNegative = false);
    static primitive loadHex(const char *&c, const bool isNegative = false);

    inline primitive& operator = (const uint8_t value_) {
      type = uint8_;
      value.uint8_ = (uint8_t) value_;
      return *this;
    }

    inline primitive& operator = (const uint16_t value_) {
      type = uint16_;
      value.uint16_ = (uint16_t) value_;
      return *this;
    }

    inline primitive& operator = (const uint32_t value_) {
      type = uint32_;
      value.uint32_ = (uint32_t) value_;
      return *this;
    }

    inline primitive& operator = (const uint64_t value_) {
      type = uint64_;
      value.uint64_ = (uint64_t) value_;
      return *this;
    }

    inline primitive& operator = (const int8_t value_) {
      type = int8_;
      value.int8_ = (int8_t) value_;
      return *this;
    }

    inline primitive& operator = (const int16_t value_) {
      type = int16_;
      value.int16_ = (int16_t) value_;
      return *this;
    }

    inline primitive& operator = (const int32_t value_) {
      type = int32_;
      value.int32_ = (int32_t) value_;
      return *this;
    }

    inline primitive& operator = (const int64_t value_) {
      type = int64_;
      value.int64_ = (int64_t) value_;
      return *this;
    }

    inline primitive& operator = (const float value_) {
      type = float_;
      value.float_ = value_;
      return *this;
    }

    inline primitive& operator = (const double value_) {
      type = double_;
      value.double_ = value_;
      return *this;
    }

    inline primitive& operator = (void *value_) {
      type = ptr;
      value.ptr = (char*) value_;
      return *this;
    }

    inline operator bool () const {
      return to<uint8_t>();
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

    template <class TM>
    inline TM to() const {
      switch(type) {
      case uint8_  : return (TM) value.uint8_;
      case uint16_ : return (TM) value.uint16_;
      case uint32_ : return (TM) value.uint32_;
      case uint64_ : return (TM) value.uint64_;
      case int8_   : return (TM) value.int8_;
      case int16_  : return (TM) value.int16_;
      case int32_  : return (TM) value.int32_;
      case int64_  : return (TM) value.int64_;
      case float_  : return (TM) value.float_;
      case double_ : return (TM) value.double_;
      default: OCCA_FORCE_ERROR("Type not set");
      }
      return TM();
    }

    inline bool isSigned() const {
      return type & isSigned_;
    }

    inline bool isUnsigned() const {
      return type & isUnsigned_;
    }

    std::string toString() const;
    operator std::string () const;

    friend std::ostream& operator << (std::ostream &out, const primitive &p);
  };

  //---[ Unary Operators ]--------------
  inline primitive not_(const primitive &p) {
    switch(p.type) {
    case primitive::int8_   : return primitive(!p.value.int8_);
    case primitive::uint8_  : return primitive(!p.value.uint8_);
    case primitive::int16_  : return primitive(!p.value.int16_);
    case primitive::uint16_ : return primitive(!p.value.uint16_);
    case primitive::int32_  : return primitive(!p.value.int32_);
    case primitive::uint32_ : return primitive(!p.value.uint32_);
    case primitive::int64_  : return primitive(!p.value.int64_);
    case primitive::uint64_ : return primitive(!p.value.uint64_);
    case primitive::float_  : return primitive(!p.value.float_);
    case primitive::double_ : return primitive(!p.value.double_);
    default: ;
    }
    return primitive();
  }

  inline primitive positive(const primitive &p) {
    switch(p.type) {
    case primitive::int8_   : return primitive(+p.value.int8_);
    case primitive::uint8_  : return primitive(+p.value.uint8_);
    case primitive::int16_  : return primitive(+p.value.int16_);
    case primitive::uint16_ : return primitive(+p.value.uint16_);
    case primitive::int32_  : return primitive(+p.value.int32_);
    case primitive::uint32_ : return primitive(+p.value.uint32_);
    case primitive::int64_  : return primitive(+p.value.int64_);
    case primitive::uint64_ : return primitive(+p.value.uint64_);
    case primitive::float_  : return primitive(+p.value.float_);
    case primitive::double_ : return primitive(+p.value.double_);
    default: ;
    }
    return primitive();
  }

  inline primitive negative(const primitive &p) {
    switch(p.type) {
    case primitive::int8_   : return primitive(-p.value.int8_);
    case primitive::uint8_  : return primitive(-p.value.uint8_);
    case primitive::int16_  : return primitive(-p.value.int16_);
    case primitive::uint16_ : return primitive(-p.value.uint16_);
    case primitive::int32_  : return primitive(-p.value.int32_);
    case primitive::uint32_ : return primitive(-p.value.uint32_);
    case primitive::int64_  : return primitive(-p.value.int64_);
    case primitive::uint64_ : return primitive(-p.value.uint64_);
    case primitive::float_  : return primitive(-p.value.float_);
    case primitive::double_ : return primitive(-p.value.double_);
    default: ;
    }
    return primitive();
  }

  inline primitive tilde(const primitive &p) {
    switch(p.type) {
    case primitive::int8_   : return primitive(~p.value.int8_);
    case primitive::uint8_  : return primitive(~p.value.uint8_);
    case primitive::int16_  : return primitive(~p.value.int16_);
    case primitive::uint16_ : return primitive(~p.value.uint16_);
    case primitive::int32_  : return primitive(~p.value.int32_);
    case primitive::uint32_ : return primitive(~p.value.uint32_);
    case primitive::int64_  : return primitive(~p.value.int64_);
    case primitive::uint64_ : return primitive(~p.value.uint64_);
    case primitive::float_  : OCCA_FORCE_ERROR("Cannot apply operator ~ to float type");   break;
    case primitive::double_ : OCCA_FORCE_ERROR("Cannot apply operator ~ to double type");  break;
    default: ;
    }
    return primitive();
  }

  inline primitive& leftIncrement(primitive &p) {
    switch(p.type) {
    case primitive::int8_   : ++p.value.int8_;    return p;
    case primitive::uint8_  : ++p.value.uint8_;   return p;
    case primitive::int16_  : ++p.value.int16_;   return p;
    case primitive::uint16_ : ++p.value.uint16_;  return p;
    case primitive::int32_  : ++p.value.int32_;   return p;
    case primitive::uint32_ : ++p.value.uint32_;  return p;
    case primitive::int64_  : ++p.value.int64_;   return p;
    case primitive::uint64_ : ++p.value.uint64_;  return p;
    case primitive::float_  : ++p.value.float_;   return p;
    case primitive::double_ : ++p.value.double_;  return p;
    default: ;
    }
    return p;
  }

  inline primitive& leftDecrement(primitive &p) {
    switch(p.type) {
    case primitive::int8_   : --p.value.int8_;    return p;
    case primitive::uint8_  : --p.value.uint8_;   return p;
    case primitive::int16_  : --p.value.int16_;   return p;
    case primitive::uint16_ : --p.value.uint16_;  return p;
    case primitive::int32_  : --p.value.int32_;   return p;
    case primitive::uint32_ : --p.value.uint32_;  return p;
    case primitive::int64_  : --p.value.int64_;   return p;
    case primitive::uint64_ : --p.value.uint64_;  return p;
    case primitive::float_  : --p.value.float_;   return p;
    case primitive::double_ : --p.value.double_;  return p;
    default: ;
    }
    return p;
  }

  inline primitive rightIncrement(primitive &p, int) {
    switch(p.type) {
    case primitive::int8_   : p.value.int8_++;    return p;
    case primitive::uint8_  : p.value.uint8_++;   return p;
    case primitive::int16_  : p.value.int16_++;   return p;
    case primitive::uint16_ : p.value.uint16_++;  return p;
    case primitive::int32_  : p.value.int32_++;   return p;
    case primitive::uint32_ : p.value.uint32_++;  return p;
    case primitive::int64_  : p.value.int64_++;   return p;
    case primitive::uint64_ : p.value.uint64_++;  return p;
    case primitive::float_  : p.value.float_++;   return p;
    case primitive::double_ : p.value.double_++;  return p;
    default: ;
    }
    return p;
  }

  inline primitive rightDecrement(primitive &p, int) {
    switch(p.type) {
    case primitive::int8_   : p.value.int8_--;    return p;
    case primitive::uint8_  : p.value.uint8_--;   return p;
    case primitive::int16_  : p.value.int16_--;   return p;
    case primitive::uint16_ : p.value.uint16_--;  return p;
    case primitive::int32_  : p.value.int32_--;   return p;
    case primitive::uint32_ : p.value.uint32_--;  return p;
    case primitive::int64_  : p.value.int64_--;   return p;
    case primitive::uint64_ : p.value.uint64_--;  return p;
    case primitive::float_  : p.value.float_--;   return p;
    case primitive::double_ : p.value.double_--;  return p;
    default: ;
    }
    return p;
  }
  //====================================


  //---[ Boolean Operators ]------------
  inline primitive lessThan(const primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : return primitive(a.to<int8_t>()   < b.to<int8_t>());
    case primitive::uint8_  : return primitive(a.to<uint8_t>()  < b.to<uint8_t>());
    case primitive::int16_  : return primitive(a.to<int16_t>()  < b.to<int16_t>());
    case primitive::uint16_ : return primitive(a.to<uint16_t>() < b.to<uint16_t>());
    case primitive::int32_  : return primitive(a.to<int32_t>()  < b.to<int32_t>());
    case primitive::uint32_ : return primitive(a.to<uint32_t>() < b.to<uint32_t>());
    case primitive::int64_  : return primitive(a.to<int64_t>()  < b.to<int64_t>());
    case primitive::uint64_ : return primitive(a.to<uint64_t>() < b.to<uint64_t>());
    case primitive::float_  : return primitive(a.to<float>()    < b.to<float>());
    case primitive::double_ : return primitive(a.to<double>()   < b.to<double>());
    default: ;
    }
    return primitive();
  }

  inline primitive lessThanEq(const primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : return primitive(a.to<int8_t>()   <= b.to<int8_t>());
    case primitive::uint8_  : return primitive(a.to<uint8_t>()  <= b.to<uint8_t>());
    case primitive::int16_  : return primitive(a.to<int16_t>()  <= b.to<int16_t>());
    case primitive::uint16_ : return primitive(a.to<uint16_t>() <= b.to<uint16_t>());
    case primitive::int32_  : return primitive(a.to<int32_t>()  <= b.to<int32_t>());
    case primitive::uint32_ : return primitive(a.to<uint32_t>() <= b.to<uint32_t>());
    case primitive::int64_  : return primitive(a.to<int64_t>()  <= b.to<int64_t>());
    case primitive::uint64_ : return primitive(a.to<uint64_t>() <= b.to<uint64_t>());
    case primitive::float_  : return primitive(a.to<float>()    <= b.to<float>());
    case primitive::double_ : return primitive(a.to<double>()   <= b.to<double>());
    default: ;
    }
    return primitive();
  }

  inline primitive equal(const primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : return primitive(a.to<int8_t>()   == b.to<int8_t>());
    case primitive::uint8_  : return primitive(a.to<uint8_t>()  == b.to<uint8_t>());
    case primitive::int16_  : return primitive(a.to<int16_t>()  == b.to<int16_t>());
    case primitive::uint16_ : return primitive(a.to<uint16_t>() == b.to<uint16_t>());
    case primitive::int32_  : return primitive(a.to<int32_t>()  == b.to<int32_t>());
    case primitive::uint32_ : return primitive(a.to<uint32_t>() == b.to<uint32_t>());
    case primitive::int64_  : return primitive(a.to<int64_t>()  == b.to<int64_t>());
    case primitive::uint64_ : return primitive(a.to<uint64_t>() == b.to<uint64_t>());
    case primitive::float_  : return primitive(a.to<float>()    == b.to<float>());
    case primitive::double_ : return primitive(a.to<double>()   == b.to<double>());
    default: ;
    }
    return primitive();
  }

  inline primitive notEqual(const primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : return primitive(a.to<int8_t>()   != b.to<int8_t>());
    case primitive::uint8_  : return primitive(a.to<uint8_t>()  != b.to<uint8_t>());
    case primitive::int16_  : return primitive(a.to<int16_t>()  != b.to<int16_t>());
    case primitive::uint16_ : return primitive(a.to<uint16_t>() != b.to<uint16_t>());
    case primitive::int32_  : return primitive(a.to<int32_t>()  != b.to<int32_t>());
    case primitive::uint32_ : return primitive(a.to<uint32_t>() != b.to<uint32_t>());
    case primitive::int64_  : return primitive(a.to<int64_t>()  != b.to<int64_t>());
    case primitive::uint64_ : return primitive(a.to<uint64_t>() != b.to<uint64_t>());
    case primitive::float_  : return primitive(a.to<float>()    != b.to<float>());
    case primitive::double_ : return primitive(a.to<double>()   != b.to<double>());
    default: ;
    }
    return primitive();
  }

  inline primitive greaterThanEq(const primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : return primitive(a.to<int8_t>()   >= b.to<int8_t>());
    case primitive::uint8_  : return primitive(a.to<uint8_t>()  >= b.to<uint8_t>());
    case primitive::int16_  : return primitive(a.to<int16_t>()  >= b.to<int16_t>());
    case primitive::uint16_ : return primitive(a.to<uint16_t>() >= b.to<uint16_t>());
    case primitive::int32_  : return primitive(a.to<int32_t>()  >= b.to<int32_t>());
    case primitive::uint32_ : return primitive(a.to<uint32_t>() >= b.to<uint32_t>());
    case primitive::int64_  : return primitive(a.to<int64_t>()  >= b.to<int64_t>());
    case primitive::uint64_ : return primitive(a.to<uint64_t>() >= b.to<uint64_t>());
    case primitive::float_  : return primitive(a.to<float>()    >= b.to<float>());
    case primitive::double_ : return primitive(a.to<double>()   >= b.to<double>());
    default: ;
    }
    return primitive();
  }

  inline primitive greaterThan(const primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : return primitive(a.to<int8_t>()   > b.to<int8_t>());
    case primitive::uint8_  : return primitive(a.to<uint8_t>()  > b.to<uint8_t>());
    case primitive::int16_  : return primitive(a.to<int16_t>()  > b.to<int16_t>());
    case primitive::uint16_ : return primitive(a.to<uint16_t>() > b.to<uint16_t>());
    case primitive::int32_  : return primitive(a.to<int32_t>()  > b.to<int32_t>());
    case primitive::uint32_ : return primitive(a.to<uint32_t>() > b.to<uint32_t>());
    case primitive::int64_  : return primitive(a.to<int64_t>()  > b.to<int64_t>());
    case primitive::uint64_ : return primitive(a.to<uint64_t>() > b.to<uint64_t>());
    case primitive::float_  : return primitive(a.to<float>()    > b.to<float>());
    case primitive::double_ : return primitive(a.to<double>()   > b.to<double>());
    default: ;
    }
    return primitive();
  }

  inline primitive and_(const primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : return primitive(a.to<int8_t>()   && b.to<int8_t>());
    case primitive::uint8_  : return primitive(a.to<uint8_t>()  && b.to<uint8_t>());
    case primitive::int16_  : return primitive(a.to<int16_t>()  && b.to<int16_t>());
    case primitive::uint16_ : return primitive(a.to<uint16_t>() && b.to<uint16_t>());
    case primitive::int32_  : return primitive(a.to<int32_t>()  && b.to<int32_t>());
    case primitive::uint32_ : return primitive(a.to<uint32_t>() && b.to<uint32_t>());
    case primitive::int64_  : return primitive(a.to<int64_t>()  && b.to<int64_t>());
    case primitive::uint64_ : return primitive(a.to<uint64_t>() && b.to<uint64_t>());
    case primitive::float_  : return primitive(a.to<float>()    && b.to<float>());
    case primitive::double_ : return primitive(a.to<double>()   && b.to<double>());
    default: ;
    }
    return primitive();
  }

  inline primitive or_(const primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : return primitive(a.to<int8_t>()   || b.to<int8_t>());
    case primitive::uint8_  : return primitive(a.to<uint8_t>()  || b.to<uint8_t>());
    case primitive::int16_  : return primitive(a.to<int16_t>()  || b.to<int16_t>());
    case primitive::uint16_ : return primitive(a.to<uint16_t>() || b.to<uint16_t>());
    case primitive::int32_  : return primitive(a.to<int32_t>()  || b.to<int32_t>());
    case primitive::uint32_ : return primitive(a.to<uint32_t>() || b.to<uint32_t>());
    case primitive::int64_  : return primitive(a.to<int64_t>()  || b.to<int64_t>());
    case primitive::uint64_ : return primitive(a.to<uint64_t>() || b.to<uint64_t>());
    case primitive::float_  : return primitive(a.to<float>()    || b.to<float>());
    case primitive::double_ : return primitive(a.to<double>()   || b.to<double>());
    default: ;
    }
    return primitive();
  }
  //====================================


  //---[ Binary Operators ]-------------
  inline primitive mult(const primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : return primitive(a.to<int8_t>()   * b.to<int8_t>());
    case primitive::uint8_  : return primitive(a.to<uint8_t>()  * b.to<uint8_t>());
    case primitive::int16_  : return primitive(a.to<int16_t>()  * b.to<int16_t>());
    case primitive::uint16_ : return primitive(a.to<uint16_t>() * b.to<uint16_t>());
    case primitive::int32_  : return primitive(a.to<int32_t>()  * b.to<int32_t>());
    case primitive::uint32_ : return primitive(a.to<uint32_t>() * b.to<uint32_t>());
    case primitive::int64_  : return primitive(a.to<int64_t>()  * b.to<int64_t>());
    case primitive::uint64_ : return primitive(a.to<uint64_t>() * b.to<uint64_t>());
    case primitive::float_  : return primitive(a.to<float>()    * b.to<float>());
    case primitive::double_ : return primitive(a.to<double>()   * b.to<double>());
    default: ;
    }
    return primitive();
  }

  inline primitive add(const primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : return primitive(a.to<int8_t>()   + b.to<int8_t>());
    case primitive::uint8_  : return primitive(a.to<uint8_t>()  + b.to<uint8_t>());
    case primitive::int16_  : return primitive(a.to<int16_t>()  + b.to<int16_t>());
    case primitive::uint16_ : return primitive(a.to<uint16_t>() + b.to<uint16_t>());
    case primitive::int32_  : return primitive(a.to<int32_t>()  + b.to<int32_t>());
    case primitive::uint32_ : return primitive(a.to<uint32_t>() + b.to<uint32_t>());
    case primitive::int64_  : return primitive(a.to<int64_t>()  + b.to<int64_t>());
    case primitive::uint64_ : return primitive(a.to<uint64_t>() + b.to<uint64_t>());
    case primitive::float_  : return primitive(a.to<float>()    + b.to<float>());
    case primitive::double_ : return primitive(a.to<double>()   + b.to<double>());
    default: ;
    }
    return primitive();
  }

  inline primitive sub(const primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : return primitive(a.to<int8_t>()   - b.to<int8_t>());
    case primitive::uint8_  : return primitive(a.to<uint8_t>()  - b.to<uint8_t>());
    case primitive::int16_  : return primitive(a.to<int16_t>()  - b.to<int16_t>());
    case primitive::uint16_ : return primitive(a.to<uint16_t>() - b.to<uint16_t>());
    case primitive::int32_  : return primitive(a.to<int32_t>()  - b.to<int32_t>());
    case primitive::uint32_ : return primitive(a.to<uint32_t>() - b.to<uint32_t>());
    case primitive::int64_  : return primitive(a.to<int64_t>()  - b.to<int64_t>());
    case primitive::uint64_ : return primitive(a.to<uint64_t>() - b.to<uint64_t>());
    case primitive::float_  : return primitive(a.to<float>()    - b.to<float>());
    case primitive::double_ : return primitive(a.to<double>()   - b.to<double>());
    default: ;
    }
    return primitive();
  }

  inline primitive div(const primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : return primitive(a.to<int8_t>()   / b.to<int8_t>());
    case primitive::uint8_  : return primitive(a.to<uint8_t>()  / b.to<uint8_t>());
    case primitive::int16_  : return primitive(a.to<int16_t>()  / b.to<int16_t>());
    case primitive::uint16_ : return primitive(a.to<uint16_t>() / b.to<uint16_t>());
    case primitive::int32_  : return primitive(a.to<int32_t>()  / b.to<int32_t>());
    case primitive::uint32_ : return primitive(a.to<uint32_t>() / b.to<uint32_t>());
    case primitive::int64_  : return primitive(a.to<int64_t>()  / b.to<int64_t>());
    case primitive::uint64_ : return primitive(a.to<uint64_t>() / b.to<uint64_t>());
    case primitive::float_  : return primitive(a.to<float>()    / b.to<float>());
    case primitive::double_ : return primitive(a.to<double>()   / b.to<double>());
    default: ;
    }
    return primitive();
  }

  inline primitive mod(const primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : return primitive(a.to<int8_t>()   % b.to<int8_t>());
    case primitive::uint8_  : return primitive(a.to<uint8_t>()  % b.to<uint8_t>());
    case primitive::int16_  : return primitive(a.to<int16_t>()  % b.to<int16_t>());
    case primitive::uint16_ : return primitive(a.to<uint16_t>() % b.to<uint16_t>());
    case primitive::int32_  : return primitive(a.to<int32_t>()  % b.to<int32_t>());
    case primitive::uint32_ : return primitive(a.to<uint32_t>() % b.to<uint32_t>());
    case primitive::int64_  : return primitive(a.to<int64_t>()  % b.to<int64_t>());
    case primitive::uint64_ : return primitive(a.to<uint64_t>() % b.to<uint64_t>());
    case primitive::float_  : OCCA_FORCE_ERROR("Cannot apply operator % to float type"); break;
    case primitive::double_ : OCCA_FORCE_ERROR("Cannot apply operator % to double type"); break;
    default: ;
    }
    return primitive();
  }

  inline primitive bitAnd(const primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : return primitive(a.to<int8_t>()   & b.to<int8_t>());
    case primitive::uint8_  : return primitive(a.to<uint8_t>()  & b.to<uint8_t>());
    case primitive::int16_  : return primitive(a.to<int16_t>()  & b.to<int16_t>());
    case primitive::uint16_ : return primitive(a.to<uint16_t>() & b.to<uint16_t>());
    case primitive::int32_  : return primitive(a.to<int32_t>()  & b.to<int32_t>());
    case primitive::uint32_ : return primitive(a.to<uint32_t>() & b.to<uint32_t>());
    case primitive::int64_  : return primitive(a.to<int64_t>()  & b.to<int64_t>());
    case primitive::uint64_ : return primitive(a.to<uint64_t>() & b.to<uint64_t>());
    case primitive::float_  : OCCA_FORCE_ERROR("Cannot apply operator & to float type");   break;
    case primitive::double_ : OCCA_FORCE_ERROR("Cannot apply operator & to double type");  break;
    default: ;
    }
    return primitive();
  }

  inline primitive bitOr(const primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : return primitive(a.to<int8_t>()   | b.to<int8_t>());
    case primitive::uint8_  : return primitive(a.to<uint8_t>()  | b.to<uint8_t>());
    case primitive::int16_  : return primitive(a.to<int16_t>()  | b.to<int16_t>());
    case primitive::uint16_ : return primitive(a.to<uint16_t>() | b.to<uint16_t>());
    case primitive::int32_  : return primitive(a.to<int32_t>()  | b.to<int32_t>());
    case primitive::uint32_ : return primitive(a.to<uint32_t>() | b.to<uint32_t>());
    case primitive::int64_  : return primitive(a.to<int64_t>()  | b.to<int64_t>());
    case primitive::uint64_ : return primitive(a.to<uint64_t>() | b.to<uint64_t>());
    case primitive::float_  : OCCA_FORCE_ERROR("Cannot apply operator | to float type");   break;
    case primitive::double_ : OCCA_FORCE_ERROR("Cannot apply operator | to double type");  break;
    default: ;
    }
    return primitive();
  }

  inline primitive xor_(const primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : return primitive(a.to<int8_t>()   ^ b.to<int8_t>());
    case primitive::uint8_  : return primitive(a.to<uint8_t>()  ^ b.to<uint8_t>());
    case primitive::int16_  : return primitive(a.to<int16_t>()  ^ b.to<int16_t>());
    case primitive::uint16_ : return primitive(a.to<uint16_t>() ^ b.to<uint16_t>());
    case primitive::int32_  : return primitive(a.to<int32_t>()  ^ b.to<int32_t>());
    case primitive::uint32_ : return primitive(a.to<uint32_t>() ^ b.to<uint32_t>());
    case primitive::int64_  : return primitive(a.to<int64_t>()  ^ b.to<int64_t>());
    case primitive::uint64_ : return primitive(a.to<uint64_t>() ^ b.to<uint64_t>());
    case primitive::float_  : OCCA_FORCE_ERROR("Cannot apply operator ^ to float type");   break;
    case primitive::double_ : OCCA_FORCE_ERROR("Cannot apply operator ^ to double type");  break;
    default: ;
    }
    return primitive();
  }

  inline primitive rightShift(const primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : return primitive(a.to<int8_t>()   >> b.to<int8_t>());
    case primitive::uint8_  : return primitive(a.to<uint8_t>()  >> b.to<uint8_t>());
    case primitive::int16_  : return primitive(a.to<int16_t>()  >> b.to<int16_t>());
    case primitive::uint16_ : return primitive(a.to<uint16_t>() >> b.to<uint16_t>());
    case primitive::int32_  : return primitive(a.to<int32_t>()  >> b.to<int32_t>());
    case primitive::uint32_ : return primitive(a.to<uint32_t>() >> b.to<uint32_t>());
    case primitive::int64_  : return primitive(a.to<int64_t>()  >> b.to<int64_t>());
    case primitive::uint64_ : return primitive(a.to<uint64_t>() >> b.to<uint64_t>());
    case primitive::float_  : OCCA_FORCE_ERROR("Cannot apply operator >> to float type");   break;
    case primitive::double_ : OCCA_FORCE_ERROR("Cannot apply operator >> to double type");  break;
    default: ;
    }
    return primitive();
  }

  inline primitive leftShift(const primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : return primitive(a.to<int8_t>()   << b.to<int8_t>());
    case primitive::uint8_  : return primitive(a.to<uint8_t>()  << b.to<uint8_t>());
    case primitive::int16_  : return primitive(a.to<int16_t>()  << b.to<int16_t>());
    case primitive::uint16_ : return primitive(a.to<uint16_t>() << b.to<uint16_t>());
    case primitive::int32_  : return primitive(a.to<int32_t>()  << b.to<int32_t>());
    case primitive::uint32_ : return primitive(a.to<uint32_t>() << b.to<uint32_t>());
    case primitive::int64_  : return primitive(a.to<int64_t>()  << b.to<int64_t>());
    case primitive::uint64_ : return primitive(a.to<uint64_t>() << b.to<uint64_t>());
    case primitive::float_  : OCCA_FORCE_ERROR("Cannot apply operator << to float type");   break;
    case primitive::double_ : OCCA_FORCE_ERROR("Cannot apply operator << to double type");  break;
    default: ;
    }
    return primitive();
  }
  //====================================


  //---[ Assignment Operators ]---------
  inline primitive& multEq(primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : a = (a.to<int8_t>()   * b.to<int8_t>());   break;
    case primitive::uint8_  : a = (a.to<uint8_t>()  * b.to<uint8_t>());  break;
    case primitive::int16_  : a = (a.to<int16_t>()  * b.to<int16_t>());  break;
    case primitive::uint16_ : a = (a.to<uint16_t>() * b.to<uint16_t>()); break;
    case primitive::int32_  : a = (a.to<int32_t>()  * b.to<int32_t>());  break;
    case primitive::uint32_ : a = (a.to<uint32_t>() * b.to<uint32_t>()); break;
    case primitive::int64_  : a = (a.to<int64_t>()  * b.to<int64_t>());  break;
    case primitive::uint64_ : a = (a.to<uint64_t>() * b.to<uint64_t>()); break;
    case primitive::float_  : a = (a.to<float>()    * b.to<float>());    break;
    case primitive::double_ : a = (a.to<double>()   * b.to<double>());   break;
    default: ;
    }
    return a;
  }

  inline primitive& addEq(primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : a = (a.to<int8_t>()   + b.to<int8_t>());   break;
    case primitive::uint8_  : a = (a.to<uint8_t>()  + b.to<uint8_t>());  break;
    case primitive::int16_  : a = (a.to<int16_t>()  + b.to<int16_t>());  break;
    case primitive::uint16_ : a = (a.to<uint16_t>() + b.to<uint16_t>()); break;
    case primitive::int32_  : a = (a.to<int32_t>()  + b.to<int32_t>());  break;
    case primitive::uint32_ : a = (a.to<uint32_t>() + b.to<uint32_t>()); break;
    case primitive::int64_  : a = (a.to<int64_t>()  + b.to<int64_t>());  break;
    case primitive::uint64_ : a = (a.to<uint64_t>() + b.to<uint64_t>()); break;
    case primitive::float_  : a = (a.to<float>()    + b.to<float>());    break;
    case primitive::double_ : a = (a.to<double>()   + b.to<double>());   break;
    default: ;
    }
    return a;
  }

  inline primitive& subEq(primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : a = (a.to<int8_t>()   - b.to<int8_t>());   break;
    case primitive::uint8_  : a = (a.to<uint8_t>()  - b.to<uint8_t>());  break;
    case primitive::int16_  : a = (a.to<int16_t>()  - b.to<int16_t>());  break;
    case primitive::uint16_ : a = (a.to<uint16_t>() - b.to<uint16_t>()); break;
    case primitive::int32_  : a = (a.to<int32_t>()  - b.to<int32_t>());  break;
    case primitive::uint32_ : a = (a.to<uint32_t>() - b.to<uint32_t>()); break;
    case primitive::int64_  : a = (a.to<int64_t>()  - b.to<int64_t>());  break;
    case primitive::uint64_ : a = (a.to<uint64_t>() - b.to<uint64_t>()); break;
    case primitive::float_  : a = (a.to<float>()    - b.to<float>());    break;
    case primitive::double_ : a = (a.to<double>()   - b.to<double>());   break;
    default: ;
    }
    return a;
  }

  inline primitive& divEq(primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : a = (a.to<int8_t>()   / b.to<int8_t>());   break;
    case primitive::uint8_  : a = (a.to<uint8_t>()  / b.to<uint8_t>());  break;
    case primitive::int16_  : a = (a.to<int16_t>()  / b.to<int16_t>());  break;
    case primitive::uint16_ : a = (a.to<uint16_t>() / b.to<uint16_t>()); break;
    case primitive::int32_  : a = (a.to<int32_t>()  / b.to<int32_t>());  break;
    case primitive::uint32_ : a = (a.to<uint32_t>() / b.to<uint32_t>()); break;
    case primitive::int64_  : a = (a.to<int64_t>()  / b.to<int64_t>());  break;
    case primitive::uint64_ : a = (a.to<uint64_t>() / b.to<uint64_t>()); break;
    case primitive::float_  : a = (a.to<float>()    / b.to<float>());    break;
    case primitive::double_ : a = (a.to<double>()   / b.to<double>());   break;
    default: ;
    }
    return a;
  }

  inline primitive& modEq(primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : a = (a.to<int8_t>()   % b.to<int8_t>());   break;
    case primitive::uint8_  : a = (a.to<uint8_t>()  % b.to<uint8_t>());  break;
    case primitive::int16_  : a = (a.to<int16_t>()  % b.to<int16_t>());  break;
    case primitive::uint16_ : a = (a.to<uint16_t>() % b.to<uint16_t>()); break;
    case primitive::int32_  : a = (a.to<int32_t>()  % b.to<int32_t>());  break;
    case primitive::uint32_ : a = (a.to<uint32_t>() % b.to<uint32_t>()); break;
    case primitive::int64_  : a = (a.to<int64_t>()  % b.to<int64_t>());  break;
    case primitive::uint64_ : a = (a.to<uint64_t>() % b.to<uint64_t>()); break;
    case primitive::float_  : OCCA_FORCE_ERROR("Cannot apply operator % to float type"); break;
    case primitive::double_ : OCCA_FORCE_ERROR("Cannot apply operator % to double type"); break;
    default: ;
    }
    return a;
  }

  inline primitive& bitAndEq(primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : a = (a.to<int8_t>()   & b.to<int8_t>());   break;
    case primitive::uint8_  : a = (a.to<uint8_t>()  & b.to<uint8_t>());  break;
    case primitive::int16_  : a = (a.to<int16_t>()  & b.to<int16_t>());  break;
    case primitive::uint16_ : a = (a.to<uint16_t>() & b.to<uint16_t>()); break;
    case primitive::int32_  : a = (a.to<int32_t>()  & b.to<int32_t>());  break;
    case primitive::uint32_ : a = (a.to<uint32_t>() & b.to<uint32_t>()); break;
    case primitive::int64_  : a = (a.to<int64_t>()  & b.to<int64_t>());  break;
    case primitive::uint64_ : a = (a.to<uint64_t>() & b.to<uint64_t>()); break;
    case primitive::float_  : OCCA_FORCE_ERROR("Cannot apply operator & to float type");  break;
    case primitive::double_ : OCCA_FORCE_ERROR("Cannot apply operator & to double type"); break;
    default: ;
    }
    return a;
  }

  inline primitive& bitOrEq(primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : a = (a.to<int8_t>()   | b.to<int8_t>());   break;
    case primitive::uint8_  : a = (a.to<uint8_t>()  | b.to<uint8_t>());  break;
    case primitive::int16_  : a = (a.to<int16_t>()  | b.to<int16_t>());  break;
    case primitive::uint16_ : a = (a.to<uint16_t>() | b.to<uint16_t>()); break;
    case primitive::int32_  : a = (a.to<int32_t>()  | b.to<int32_t>());  break;
    case primitive::uint32_ : a = (a.to<uint32_t>() | b.to<uint32_t>()); break;
    case primitive::int64_  : a = (a.to<int64_t>()  | b.to<int64_t>());  break;
    case primitive::uint64_ : a = (a.to<uint64_t>() | b.to<uint64_t>()); break;
    case primitive::float_  : OCCA_FORCE_ERROR("Cannot apply operator | to float type");  break;
    case primitive::double_ : OCCA_FORCE_ERROR("Cannot apply operator | to double type"); break;
    default: ;
    }
    return a;
  }

  inline primitive& xor_Eq(primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : a = (a.to<int8_t>()   ^ b.to<int8_t>());   break;
    case primitive::uint8_  : a = (a.to<uint8_t>()  ^ b.to<uint8_t>());  break;
    case primitive::int16_  : a = (a.to<int16_t>()  ^ b.to<int16_t>());  break;
    case primitive::uint16_ : a = (a.to<uint16_t>() ^ b.to<uint16_t>()); break;
    case primitive::int32_  : a = (a.to<int32_t>()  ^ b.to<int32_t>());  break;
    case primitive::uint32_ : a = (a.to<uint32_t>() ^ b.to<uint32_t>()); break;
    case primitive::int64_  : a = (a.to<int64_t>()  ^ b.to<int64_t>());  break;
    case primitive::uint64_ : a = (a.to<uint64_t>() ^ b.to<uint64_t>()); break;
    case primitive::float_  : OCCA_FORCE_ERROR("Cannot apply operator ^ to float type");  break;
    case primitive::double_ : OCCA_FORCE_ERROR("Cannot apply operator ^ to double type"); break;
    default: ;
    }
    return a;
  }

  inline primitive& rightShiftEq(primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : a = (a.to<int8_t>()   >> b.to<int8_t>());   break;
    case primitive::uint8_  : a = (a.to<uint8_t>()  >> b.to<uint8_t>());  break;
    case primitive::int16_  : a = (a.to<int16_t>()  >> b.to<int16_t>());  break;
    case primitive::uint16_ : a = (a.to<uint16_t>() >> b.to<uint16_t>()); break;
    case primitive::int32_  : a = (a.to<int32_t>()  >> b.to<int32_t>());  break;
    case primitive::uint32_ : a = (a.to<uint32_t>() >> b.to<uint32_t>()); break;
    case primitive::int64_  : a = (a.to<int64_t>()  >> b.to<int64_t>());  break;
    case primitive::uint64_ : a = (a.to<uint64_t>() >> b.to<uint64_t>()); break;
    case primitive::float_  : OCCA_FORCE_ERROR("Cannot apply operator >> to float type");  break;
    case primitive::double_ : OCCA_FORCE_ERROR("Cannot apply operator >> to double type"); break;
    default: ;
    }
    return a;
  }

  inline primitive& leftShiftEq(primitive &a, const primitive &b) {
    const primitive::type_t retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
    case primitive::int8_   : a = (a.to<int8_t>()   << b.to<int8_t>());   break;
    case primitive::uint8_  : a = (a.to<uint8_t>()  << b.to<uint8_t>());  break;
    case primitive::int16_  : a = (a.to<int16_t>()  << b.to<int16_t>());  break;
    case primitive::uint16_ : a = (a.to<uint16_t>() << b.to<uint16_t>()); break;
    case primitive::int32_  : a = (a.to<int32_t>()  << b.to<int32_t>());  break;
    case primitive::uint32_ : a = (a.to<uint32_t>() << b.to<uint32_t>()); break;
    case primitive::int64_  : a = (a.to<int64_t>()  << b.to<int64_t>());  break;
    case primitive::uint64_ : a = (a.to<uint64_t>() << b.to<uint64_t>()); break;
    case primitive::float_  : OCCA_FORCE_ERROR("Cannot apply operator << to float type");  break;
    case primitive::double_ : OCCA_FORCE_ERROR("Cannot apply operator << to double type"); break;
    default: ;
    }
    return a;
  }
  //====================================
}
#endif
