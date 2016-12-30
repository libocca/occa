#ifndef OCCA_PARSER_PRIMITIVE_HEADER2
#define OCCA_PARSER_PRIMITIVE_HEADER2

#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <cstdlib>

#include "occa/defines.hpp"
#include "occa/tools/string.hpp"
#include "occa/tools/sys.hpp"

namespace occa {
  enum primitiveType_t {
    none       = (1 << 0),

    isSigned   = (0x55 << 1),
    isUnsigned = (0x55 << 2),
    isInteger  = (0xFF << 1),

    int8_      = (1 << 1),
    uint8_     = (1 << 2),
    int16_     = (1 << 3),
    uint16_    = (1 << 4),
    int32_     = (1 << 5),
    uint32_    = (1 << 6),
    int64_     = (1 << 7),
    uint64_    = (1 << 8),

    isFloat    = (3 << 9),
    float_     = (1 << 9),
    double_    = (1 << 10),

    ptr        = (1 << 11)
  };

  class primitive {
  public:
    primitiveType_t type;

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

    inline operator uint8_t () {
      return to<uint8_t>();
    }

    inline operator uint16_t () {
      return to<uint16_t>();
    }

    inline operator uint32_t () {
      return to<uint32_t>();
    }

    inline operator uint64_t () {
      return to<uint64_t>();
    }

    inline operator int8_t () {
      return to<int8_t>();
    }

    inline operator int16_t () {
      return to<int16_t>();
    }

    inline operator int32_t () {
      return to<int32_t>();
    }

    inline operator int64_t () {
      return to<int64_t>();
    }

    inline operator float () {
      return to<float>();
    }

    inline operator double () {
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
      return type & occa::isSigned;
    }

    inline bool isUnsigned() const {
      return type & occa::isUnsigned;
    }

    operator std::string () const;

    friend std::ostream& operator << (std::ostream &out, const primitive &p);

    //---[ Unary Operators ]--------------
    inline primitive operator ! () {
      switch(type) {
      case int8_   : return primitive(!value.int8_);
      case uint8_  : return primitive(!value.uint8_);
      case int16_  : return primitive(!value.int16_);
      case uint16_ : return primitive(!value.uint16_);
      case int32_  : return primitive(!value.int32_);
      case uint32_ : return primitive(!value.uint32_);
      case int64_  : return primitive(!value.int64_);
      case uint64_ : return primitive(!value.uint64_);
      case float_  : return primitive(!value.float_);
      case double_ : return primitive(!value.double_);
      default: ;
      }
      return primitive();
    }

    inline primitive operator + () {
      switch(type) {
      case int8_   : return primitive(+value.int8_);
      case uint8_  : return primitive(+value.uint8_);
      case int16_  : return primitive(+value.int16_);
      case uint16_ : return primitive(+value.uint16_);
      case int32_  : return primitive(+value.int32_);
      case uint32_ : return primitive(+value.uint32_);
      case int64_  : return primitive(+value.int64_);
      case uint64_ : return primitive(+value.uint64_);
      case float_  : return primitive(+value.float_);
      case double_ : return primitive(+value.double_);
      default: ;
      }
      return primitive();
    }

    inline primitive operator - () {
      switch(type) {
      case int8_   : return primitive(-value.int8_);
      case uint8_  : return primitive(-value.uint8_);
      case int16_  : return primitive(-value.int16_);
      case uint16_ : return primitive(-value.uint16_);
      case int32_  : return primitive(-value.int32_);
      case uint32_ : return primitive(-value.uint32_);
      case int64_  : return primitive(-value.int64_);
      case uint64_ : return primitive(-value.uint64_);
      case float_  : return primitive(-value.float_);
      case double_ : return primitive(-value.double_);
      default: ;
      }
      return primitive();
    }

    inline primitive operator ~ () {
      switch(type) {
      case int8_   : return primitive(~value.int8_);
      case uint8_  : return primitive(~value.uint8_);
      case int16_  : return primitive(~value.int16_);
      case uint16_ : return primitive(~value.uint16_);
      case int32_  : return primitive(~value.int32_);
      case uint32_ : return primitive(~value.uint32_);
      case int64_  : return primitive(~value.int64_);
      case uint64_ : return primitive(~value.uint64_);
      case float_  : OCCA_FORCE_ERROR("Cannot apply operator ~ to float type");   break;
      case double_ : OCCA_FORCE_ERROR("Cannot apply operator ~ to double type");  break;
      default: ;
      }
      return primitive();
    }

    inline primitive& operator ++ () {
      switch(type) {
      case int8_   : ++value.int8_;    return *this;
      case uint8_  : ++value.uint8_;   return *this;
      case int16_  : ++value.int16_;   return *this;
      case uint16_ : ++value.uint16_;  return *this;
      case int32_  : ++value.int32_;   return *this;
      case uint32_ : ++value.uint32_;  return *this;
      case int64_  : ++value.int64_;   return *this;
      case uint64_ : ++value.uint64_;  return *this;
      case float_  : ++value.float_;   return *this;
      case double_ : ++value.double_;  return *this;
      default: ;
      }
      return *this;
    }

    inline primitive& operator -- () {
      switch(type) {
      case int8_   : --value.int8_;    return *this;
      case uint8_  : --value.uint8_;   return *this;
      case int16_  : --value.int16_;   return *this;
      case uint16_ : --value.uint16_;  return *this;
      case int32_  : --value.int32_;   return *this;
      case uint32_ : --value.uint32_;  return *this;
      case int64_  : --value.int64_;   return *this;
      case uint64_ : --value.uint64_;  return *this;
      case float_  : --value.float_;   return *this;
      case double_ : --value.double_;  return *this;
      default: ;
      }
      return *this;
    }

    inline primitive operator ++ (int) {
      switch(type) {
      case int8_   : value.int8_++;    return *this;
      case uint8_  : value.uint8_++;   return *this;
      case int16_  : value.int16_++;   return *this;
      case uint16_ : value.uint16_++;  return *this;
      case int32_  : value.int32_++;   return *this;
      case uint32_ : value.uint32_++;  return *this;
      case int64_  : value.int64_++;   return *this;
      case uint64_ : value.uint64_++;  return *this;
      case float_  : value.float_++;   return *this;
      case double_ : value.double_++;  return *this;
      default: ;
      }
      return *this;
    }

    inline primitive operator -- (int) {
      switch(type) {
      case int8_   : value.int8_--;    return *this;
      case uint8_  : value.uint8_--;   return *this;
      case int16_  : value.int16_--;   return *this;
      case uint16_ : value.uint16_--;  return *this;
      case int32_  : value.int32_--;   return *this;
      case uint32_ : value.uint32_--;  return *this;
      case int64_  : value.int64_--;   return *this;
      case uint64_ : value.uint64_--;  return *this;
      case float_  : value.float_--;   return *this;
      case double_ : value.double_--;  return *this;
      default: ;
      }
      return *this;
    }
    //====================================


    //---[ Boolean Operators ]------------
    inline primitive operator < (const primitive &p) {
      const primitiveType_t retType = (type > p.type) ? type : p.type;
      switch(retType) {
      case int8_   : return primitive(to<int8_t>()   < p.to<int8_t>());
      case uint8_  : return primitive(to<uint8_t>()  < p.to<uint8_t>());
      case int16_  : return primitive(to<int16_t>()  < p.to<int16_t>());
      case uint16_ : return primitive(to<uint16_t>() < p.to<uint16_t>());
      case int32_  : return primitive(to<int32_t>()  < p.to<int32_t>());
      case uint32_ : return primitive(to<uint32_t>() < p.to<uint32_t>());
      case int64_  : return primitive(to<int64_t>()  < p.to<int64_t>());
      case uint64_ : return primitive(to<uint64_t>() < p.to<uint64_t>());
      case float_  : return primitive(to<float>()    < p.to<float>());
      case double_ : return primitive(to<double>()   < p.to<double>());
      default: ;
      }
      return primitive();
    }

    inline primitive operator == (const primitive &p) {
      const primitiveType_t retType = (type > p.type) ? type : p.type;
      switch(retType) {
      case int8_   : return primitive(to<int8_t>()   == p.to<int8_t>());
      case uint8_  : return primitive(to<uint8_t>()  == p.to<uint8_t>());
      case int16_  : return primitive(to<int16_t>()  == p.to<int16_t>());
      case uint16_ : return primitive(to<uint16_t>() == p.to<uint16_t>());
      case int32_  : return primitive(to<int32_t>()  == p.to<int32_t>());
      case uint32_ : return primitive(to<uint32_t>() == p.to<uint32_t>());
      case int64_  : return primitive(to<int64_t>()  == p.to<int64_t>());
      case uint64_ : return primitive(to<uint64_t>() == p.to<uint64_t>());
      case float_  : return primitive(to<float>()    == p.to<float>());
      case double_ : return primitive(to<double>()   == p.to<double>());
      default: ;
      }
      return primitive();
    }

    inline primitive operator != (const primitive &p) {
      const primitiveType_t retType = (type > p.type) ? type : p.type;
      switch(retType) {
      case int8_   : return primitive(to<int8_t>()   != p.to<int8_t>());
      case uint8_  : return primitive(to<uint8_t>()  != p.to<uint8_t>());
      case int16_  : return primitive(to<int16_t>()  != p.to<int16_t>());
      case uint16_ : return primitive(to<uint16_t>() != p.to<uint16_t>());
      case int32_  : return primitive(to<int32_t>()  != p.to<int32_t>());
      case uint32_ : return primitive(to<uint32_t>() != p.to<uint32_t>());
      case int64_  : return primitive(to<int64_t>()  != p.to<int64_t>());
      case uint64_ : return primitive(to<uint64_t>() != p.to<uint64_t>());
      case float_  : return primitive(to<float>()    != p.to<float>());
      case double_ : return primitive(to<double>()   != p.to<double>());
      default: ;
      }
      return primitive();
    }

    inline primitive operator >= (const primitive &p) {
      const primitiveType_t retType = (type > p.type) ? type : p.type;
      switch(retType) {
      case int8_   : return primitive(to<int8_t>()   >= p.to<int8_t>());
      case uint8_  : return primitive(to<uint8_t>()  >= p.to<uint8_t>());
      case int16_  : return primitive(to<int16_t>()  >= p.to<int16_t>());
      case uint16_ : return primitive(to<uint16_t>() >= p.to<uint16_t>());
      case int32_  : return primitive(to<int32_t>()  >= p.to<int32_t>());
      case uint32_ : return primitive(to<uint32_t>() >= p.to<uint32_t>());
      case int64_  : return primitive(to<int64_t>()  >= p.to<int64_t>());
      case uint64_ : return primitive(to<uint64_t>() >= p.to<uint64_t>());
      case float_  : return primitive(to<float>()    >= p.to<float>());
      case double_ : return primitive(to<double>()   >= p.to<double>());
      default: ;
      }
      return primitive();
    }

    inline primitive operator > (const primitive &p) {
      const primitiveType_t retType = (type > p.type) ? type : p.type;
      switch(retType) {
      case int8_   : return primitive(to<int8_t>()   > p.to<int8_t>());
      case uint8_  : return primitive(to<uint8_t>()  > p.to<uint8_t>());
      case int16_  : return primitive(to<int16_t>()  > p.to<int16_t>());
      case uint16_ : return primitive(to<uint16_t>() > p.to<uint16_t>());
      case int32_  : return primitive(to<int32_t>()  > p.to<int32_t>());
      case uint32_ : return primitive(to<uint32_t>() > p.to<uint32_t>());
      case int64_  : return primitive(to<int64_t>()  > p.to<int64_t>());
      case uint64_ : return primitive(to<uint64_t>() > p.to<uint64_t>());
      case float_  : return primitive(to<float>()    > p.to<float>());
      case double_ : return primitive(to<double>()   > p.to<double>());
      default: ;
      }
      return primitive();
    }

    inline primitive operator && (const primitive &p) {
      const primitiveType_t retType = (type > p.type) ? type : p.type;
      switch(retType) {
      case int8_   : return primitive(to<int8_t>()   && p.to<int8_t>());
      case uint8_  : return primitive(to<uint8_t>()  && p.to<uint8_t>());
      case int16_  : return primitive(to<int16_t>()  && p.to<int16_t>());
      case uint16_ : return primitive(to<uint16_t>() && p.to<uint16_t>());
      case int32_  : return primitive(to<int32_t>()  && p.to<int32_t>());
      case uint32_ : return primitive(to<uint32_t>() && p.to<uint32_t>());
      case int64_  : return primitive(to<int64_t>()  && p.to<int64_t>());
      case uint64_ : return primitive(to<uint64_t>() && p.to<uint64_t>());
      case float_  : return primitive(to<float>()    && p.to<float>());
      case double_ : return primitive(to<double>()   && p.to<double>());
      default: ;
      }
      return primitive();
    }

    inline primitive operator || (const primitive &p) {
      const primitiveType_t retType = (type > p.type) ? type : p.type;
      switch(retType) {
      case int8_   : return primitive(to<int8_t>()   || p.to<int8_t>());
      case uint8_  : return primitive(to<uint8_t>()  || p.to<uint8_t>());
      case int16_  : return primitive(to<int16_t>()  || p.to<int16_t>());
      case uint16_ : return primitive(to<uint16_t>() || p.to<uint16_t>());
      case int32_  : return primitive(to<int32_t>()  || p.to<int32_t>());
      case uint32_ : return primitive(to<uint32_t>() || p.to<uint32_t>());
      case int64_  : return primitive(to<int64_t>()  || p.to<int64_t>());
      case uint64_ : return primitive(to<uint64_t>() || p.to<uint64_t>());
      case float_  : return primitive(to<float>()    || p.to<float>());
      case double_ : return primitive(to<double>()   || p.to<double>());
      default: ;
      }
      return primitive();
    }
    //====================================


    //---[ Binary Operators ]-------------
    inline primitive operator * (const primitive &p) {
      const primitiveType_t retType = (type > p.type) ? type : p.type;
      switch(retType) {
      case int8_   : return primitive(to<int8_t>()   * p.to<int8_t>());
      case uint8_  : return primitive(to<uint8_t>()  * p.to<uint8_t>());
      case int16_  : return primitive(to<int16_t>()  * p.to<int16_t>());
      case uint16_ : return primitive(to<uint16_t>() * p.to<uint16_t>());
      case int32_  : return primitive(to<int32_t>()  * p.to<int32_t>());
      case uint32_ : return primitive(to<uint32_t>() * p.to<uint32_t>());
      case int64_  : return primitive(to<int64_t>()  * p.to<int64_t>());
      case uint64_ : return primitive(to<uint64_t>() * p.to<uint64_t>());
      case float_  : return primitive(to<float>()    * p.to<float>());
      case double_ : return primitive(to<double>()   * p.to<double>());
      default: ;
      }
      return primitive();
    }

    inline primitive operator + (const primitive &p) {
      const primitiveType_t retType = (type > p.type) ? type : p.type;
      switch(retType) {
      case int8_   : return primitive(to<int8_t>()   + p.to<int8_t>());
      case uint8_  : return primitive(to<uint8_t>()  + p.to<uint8_t>());
      case int16_  : return primitive(to<int16_t>()  + p.to<int16_t>());
      case uint16_ : return primitive(to<uint16_t>() + p.to<uint16_t>());
      case int32_  : return primitive(to<int32_t>()  + p.to<int32_t>());
      case uint32_ : return primitive(to<uint32_t>() + p.to<uint32_t>());
      case int64_  : return primitive(to<int64_t>()  + p.to<int64_t>());
      case uint64_ : return primitive(to<uint64_t>() + p.to<uint64_t>());
      case float_  : return primitive(to<float>()    + p.to<float>());
      case double_ : return primitive(to<double>()   + p.to<double>());
      default: ;
      }
      return primitive();
    }

    inline primitive operator - (const primitive &p) {
      const primitiveType_t retType = (type > p.type) ? type : p.type;
      switch(retType) {
      case int8_   : return primitive(to<int8_t>()   - p.to<int8_t>());
      case uint8_  : return primitive(to<uint8_t>()  - p.to<uint8_t>());
      case int16_  : return primitive(to<int16_t>()  - p.to<int16_t>());
      case uint16_ : return primitive(to<uint16_t>() - p.to<uint16_t>());
      case int32_  : return primitive(to<int32_t>()  - p.to<int32_t>());
      case uint32_ : return primitive(to<uint32_t>() - p.to<uint32_t>());
      case int64_  : return primitive(to<int64_t>()  - p.to<int64_t>());
      case uint64_ : return primitive(to<uint64_t>() - p.to<uint64_t>());
      case float_  : return primitive(to<float>()    - p.to<float>());
      case double_ : return primitive(to<double>()   - p.to<double>());
      default: ;
      }
      return primitive();
    }

    inline primitive operator / (const primitive &p) {
      const primitiveType_t retType = (type > p.type) ? type : p.type;
      switch(retType) {
      case int8_   : return primitive(to<int8_t>()   / p.to<int8_t>());
      case uint8_  : return primitive(to<uint8_t>()  / p.to<uint8_t>());
      case int16_  : return primitive(to<int16_t>()  / p.to<int16_t>());
      case uint16_ : return primitive(to<uint16_t>() / p.to<uint16_t>());
      case int32_  : return primitive(to<int32_t>()  / p.to<int32_t>());
      case uint32_ : return primitive(to<uint32_t>() / p.to<uint32_t>());
      case int64_  : return primitive(to<int64_t>()  / p.to<int64_t>());
      case uint64_ : return primitive(to<uint64_t>() / p.to<uint64_t>());
      case float_  : return primitive(to<float>()    / p.to<float>());
      case double_ : return primitive(to<double>()   / p.to<double>());
      default: ;
      }
      return primitive();
    }

    inline primitive operator % (const primitive &p) {
      const primitiveType_t retType = (type > p.type) ? type : p.type;
      switch(retType) {
      case int8_   : return primitive(to<int8_t>()   % p.to<int8_t>());
      case uint8_  : return primitive(to<uint8_t>()  % p.to<uint8_t>());
      case int16_  : return primitive(to<int16_t>()  % p.to<int16_t>());
      case uint16_ : return primitive(to<uint16_t>() % p.to<uint16_t>());
      case int32_  : return primitive(to<int32_t>()  % p.to<int32_t>());
      case uint32_ : return primitive(to<uint32_t>() % p.to<uint32_t>());
      case int64_  : return primitive(to<int64_t>()  % p.to<int64_t>());
      case uint64_ : return primitive(to<uint64_t>() % p.to<uint64_t>());
      case float_  : OCCA_FORCE_ERROR("Cannot apply operator % to pointer type"); break;
      case double_ : OCCA_FORCE_ERROR("Cannot apply operator % to pointer type"); break;
      default: ;
      }
      return primitive();
    }

    inline primitive operator & (const primitive &p) {
      const primitiveType_t retType = (type > p.type) ? type : p.type;
      switch(retType) {
      case int8_   : return primitive(to<int8_t>()   & p.to<int8_t>());
      case uint8_  : return primitive(to<uint8_t>()  & p.to<uint8_t>());
      case int16_  : return primitive(to<int16_t>()  & p.to<int16_t>());
      case uint16_ : return primitive(to<uint16_t>() & p.to<uint16_t>());
      case int32_  : return primitive(to<int32_t>()  & p.to<int32_t>());
      case uint32_ : return primitive(to<uint32_t>() & p.to<uint32_t>());
      case int64_  : return primitive(to<int64_t>()  & p.to<int64_t>());
      case uint64_ : return primitive(to<uint64_t>() & p.to<uint64_t>());
      case float_  : OCCA_FORCE_ERROR("Cannot apply operator & to float type");   break;
      case double_ : OCCA_FORCE_ERROR("Cannot apply operator & to double type");  break;
      default: ;
      }
      return primitive();
    }

    inline primitive operator | (const primitive &p) {
      const primitiveType_t retType = (type > p.type) ? type : p.type;
      switch(retType) {
      case int8_   : return primitive(to<int8_t>()   | p.to<int8_t>());
      case uint8_  : return primitive(to<uint8_t>()  | p.to<uint8_t>());
      case int16_  : return primitive(to<int16_t>()  | p.to<int16_t>());
      case uint16_ : return primitive(to<uint16_t>() | p.to<uint16_t>());
      case int32_  : return primitive(to<int32_t>()  | p.to<int32_t>());
      case uint32_ : return primitive(to<uint32_t>() | p.to<uint32_t>());
      case int64_  : return primitive(to<int64_t>()  | p.to<int64_t>());
      case uint64_ : return primitive(to<uint64_t>() | p.to<uint64_t>());
      case float_  : OCCA_FORCE_ERROR("Cannot apply operator | to float type");   break;
      case double_ : OCCA_FORCE_ERROR("Cannot apply operator | to double type");  break;
      default: ;
      }
      return primitive();
    }

    inline primitive operator ^ (const primitive &p) {
      const primitiveType_t retType = (type > p.type) ? type : p.type;
      switch(retType) {
      case int8_   : return primitive(to<int8_t>()   ^ p.to<int8_t>());
      case uint8_  : return primitive(to<uint8_t>()  ^ p.to<uint8_t>());
      case int16_  : return primitive(to<int16_t>()  ^ p.to<int16_t>());
      case uint16_ : return primitive(to<uint16_t>() ^ p.to<uint16_t>());
      case int32_  : return primitive(to<int32_t>()  ^ p.to<int32_t>());
      case uint32_ : return primitive(to<uint32_t>() ^ p.to<uint32_t>());
      case int64_  : return primitive(to<int64_t>()  ^ p.to<int64_t>());
      case uint64_ : return primitive(to<uint64_t>() ^ p.to<uint64_t>());
      case float_  : OCCA_FORCE_ERROR("Cannot apply operator ^ to float type");   break;
      case double_ : OCCA_FORCE_ERROR("Cannot apply operator ^ to double type");  break;
      default: ;
      }
      return primitive();
    }

    inline primitive operator >> (const primitive &p) {
      const primitiveType_t retType = (type > p.type) ? type : p.type;
      switch(retType) {
      case int8_   : return primitive(to<int8_t>()   >> p.to<int8_t>());
      case uint8_  : return primitive(to<uint8_t>()  >> p.to<uint8_t>());
      case int16_  : return primitive(to<int16_t>()  >> p.to<int16_t>());
      case uint16_ : return primitive(to<uint16_t>() >> p.to<uint16_t>());
      case int32_  : return primitive(to<int32_t>()  >> p.to<int32_t>());
      case uint32_ : return primitive(to<uint32_t>() >> p.to<uint32_t>());
      case int64_  : return primitive(to<int64_t>()  >> p.to<int64_t>());
      case uint64_ : return primitive(to<uint64_t>() >> p.to<uint64_t>());
      case float_  : OCCA_FORCE_ERROR("Cannot apply operator >> to float type");   break;
      case double_ : OCCA_FORCE_ERROR("Cannot apply operator >> to double type");  break;
      default: ;
      }
      return primitive();
    }

    inline primitive operator << (const primitive &p) {
      const primitiveType_t retType = (type > p.type) ? type : p.type;
      switch(retType) {
      case int8_   : return primitive(to<int8_t>()   << p.to<int8_t>());
      case uint8_  : return primitive(to<uint8_t>()  << p.to<uint8_t>());
      case int16_  : return primitive(to<int16_t>()  << p.to<int16_t>());
      case uint16_ : return primitive(to<uint16_t>() << p.to<uint16_t>());
      case int32_  : return primitive(to<int32_t>()  << p.to<int32_t>());
      case uint32_ : return primitive(to<uint32_t>() << p.to<uint32_t>());
      case int64_  : return primitive(to<int64_t>()  << p.to<int64_t>());
      case uint64_ : return primitive(to<uint64_t>() << p.to<uint64_t>());
      case float_  : OCCA_FORCE_ERROR("Cannot apply operator << to float type");   break;
      case double_ : OCCA_FORCE_ERROR("Cannot apply operator << to double type");  break;
      default: ;
      }
      return primitive();
    }
    //====================================


    //---[ Assignment Operators ]---------
    inline primitive operator *= (const primitive &p) {
      *this = (*this * p);
      return *this;
    }

    inline primitive operator += (const primitive &p) {
      *this = (*this + p);
      return *this;
    }

    inline primitive operator -= (const primitive &p) {
      *this = (*this - p);
      return *this;
    }

    inline primitive operator /= (const primitive &p) {
      *this = (*this / p);
      return *this;
    }

    inline primitive operator %= (const primitive &p) {
      *this = (*this % p);
      return *this;
    }

    inline primitive operator &= (const primitive &p) {
      *this = (*this & p);
      return *this;
    }

    inline primitive operator |= (const primitive &p) {
      *this = (*this | p);
      return *this;
    }

    inline primitive operator ^= (const primitive &p) {
      *this = (*this ^ p);
      return *this;
    }

    inline primitive operator >>= (const primitive &p) {
      *this = (*this >> p);
      return *this;
    }

    inline primitive operator <<= (const primitive &p) {
      *this = (*this << p);
      return *this;
    }
    //====================================
  };
}
#endif
