#include <cstring>

#include <occa/types/bits.hpp>
#include <occa/types/primitive.hpp>
#include <occa/internal/io.hpp>
#include <occa/internal/utils/lex.hpp>
#include <occa/internal/utils/string.hpp>

namespace occa {
  primitive::primitive(const char *c) {
    *this = load(c);
  }

  primitive::primitive(const std::string &s) {
    const char *c = s.c_str();
    *this = load(c);
  }

  primitive primitive::load(const char *&c,
                            const bool includeSign) {
    bool loadedFormattedValue = false;
    bool unsigned_ = false;
    bool negative  = false;
    bool decimal   = false;
    bool float_    = false;
    int longs      = 0;
    int digits     = 0;

    const char *c0 = c;
    primitive p;

    const int cLength = strlen(c);

    if(cLength >= 4){
      if (strncmp(c, "true", 4) == 0) {
        p = true;
        p.source = "true";

        c += 4;
        return p;
      }
    }
    if(cLength >= 5){
      if (strncmp(c, "false", 5) == 0) {
        p = false;
        p.source = "false";

        c += 5;
        return p;
      }
    }

    if ((*c == '+') || (*c == '-')) {
      if (!includeSign) {
        return primitive();
      }
      negative = (*c == '-');
      ++c;
      lex::skipWhitespace(c);
    }

    if (*c == '0') {
      ++digits;
      ++c;
      const char C = uppercase(*c);
      if ((C == 'B') || (C == 'X')) {
        loadedFormattedValue = true;

        if (C == 'B') {
          p = primitive::loadBinary(++c, negative);
        } else if (C == 'X') {
          p = primitive::loadHex(++c, negative);
        }

        if (p.type & primitiveType::none) {
          c = c0;
          return primitive();
        }
      } else {
        --c;
      }
    }

    if (!loadedFormattedValue) {
      while (true) {
        if (('0' <= *c) && (*c <= '9')) {
          ++digits;
        } else if (*c == '.') {
          decimal = true;
        } else {
          break;
        }
        ++c;
      }
    }

    if (!loadedFormattedValue && !digits) {
      c = c0;
      p.source = std::string(c0, c - c0);
      return p;
    }

    while(*c != '\0') {
      const char C = uppercase(*c);
      if (C == 'L') {
        ++longs;
        ++c;
      } else if (C == 'U') {
        unsigned_ = true;
        ++c;
      } else if (!loadedFormattedValue) {
        if (C == 'E') {
          primitive exp = primitive::load(++c);
          // Check if there was an 'F' in exp
          decimal = true;
          float_ = (exp.type & primitiveType::isFloat);
          break;
        } else if (C == 'F') {
          float_ = true;
          ++c;
        } else {
          break;
        }
      } else {
        break;
      }
    }

    if (loadedFormattedValue) {
      // Hex and binary only handle U, L, and LL
      if (longs == 0) {
        if (unsigned_) {
          p = p.to<uint32_t>();
        } else {
          p = p.to<int32_t>();
        }
      } else if (longs >= 1) {
        if (unsigned_) {
          p = p.to<uint64_t>();
        } else {
          p = p.to<int64_t>();
        }
      }
    } else {
      // Handle the multiple other formats with normal digits
      if (decimal || float_) {
        if (float_) {
          p = (float) occa::parseFloat(std::string(c0, c - c0));
        } else {
          p = (double) occa::parseDouble(std::string(c0, c - c0));
        }
      } else {
        uint64_t value_ = parseInt(std::string(c0, c - c0));
        if (longs == 0) {
          if (unsigned_) {
            p = (uint32_t) value_;
          } else {
            p = (int32_t) value_;
          }
        } else if (longs >= 1) {
          if (unsigned_) {
            p = (uint64_t) value_;
          } else {
            p = (int64_t) value_;
          }
        }
      }
    }

    p.source = std::string(c0, c - c0);
    return p;
  }

  primitive primitive::load(const std::string &s,
                            const bool includeSign) {
    const char *c = s.c_str();
    return load(c, includeSign);
  }

  primitive primitive::loadBinary(const char *&c, const bool isNegative) {
    const char *c0 = c;
    uint64_t value_ = 0;
    while (*c == '0' || *c == '1') {
      value_ = (value_ << 1) | (*c - '0');
      ++c;
    }
    if (c == c0) {
      return primitive();
    }

    const int bits = c - c0 + isNegative;
    if (bits < 8) {
      return isNegative ? primitive((int8_t) -value_) : primitive((uint8_t) value_);
    } else if (bits < 16) {
      return isNegative ? primitive((int16_t) -value_) : primitive((uint16_t) value_);
    } else if (bits < 32) {
      return isNegative ? primitive((int32_t) -value_) : primitive((uint32_t) value_);
    } else {
      return isNegative ? primitive((int64_t) -value_) : primitive((uint64_t) value_);
    }
  }

  primitive primitive::loadHex(const char *&c, const bool isNegative) {
    const char *c0 = c;
    uint64_t value_ = 0;
    while (true) {
      const char C = uppercase(*c);
      if (('0' <= C) && (C <= '9')) {
        value_ = (value_ << 4) | (C - '0');
      } else if (('A' <= C) && (C <= 'F')) {
        value_ = (value_ << 4) | (10 + C - 'A');
      } else {
        break;
      }
      ++c;
    }

    if (c == c0) {
      return primitive();
    }

    const int bits = 4*(c - c0) + isNegative;
    if (bits < 8) {
      return isNegative ? primitive((int8_t) -value_) : primitive((uint8_t) value_);
    } else if (bits < 16) {
      return isNegative ? primitive((int16_t) -value_) : primitive((uint16_t) value_);
    } else if (bits < 32) {
      return isNegative ? primitive((int32_t) -value_) : primitive((uint32_t) value_);
    } else {
      return isNegative ? primitive((int64_t) -value_) : primitive((uint64_t) value_);
    }
  }

  std::string primitive::toString() const {
    if (source.size()) {
      return source;
    }

    std::string str;
    switch (type) {
      case primitiveType::bool_   : str = (value.bool_ ? "true" : "false");         break;
      case primitiveType::uint8_  : str = occa::toString((uint64_t) value.uint8_);  break;
      case primitiveType::uint16_ : str = occa::toString((uint64_t) value.uint16_); break;
      case primitiveType::uint32_ : str = occa::toString((uint64_t) value.uint32_); break;
      case primitiveType::uint64_ : str = occa::toString((uint64_t) value.uint64_); break;
      case primitiveType::int8_   : str = occa::toString((int64_t)  value.int8_);   break;
      case primitiveType::int16_  : str = occa::toString((int64_t)  value.int16_);  break;
      case primitiveType::int32_  : str = occa::toString((int64_t)  value.int32_);  break;
      case primitiveType::int64_  : str = occa::toString((int64_t)  value.int64_);  break;
      case primitiveType::float_  : str = occa::toString(value.float_);  break;
      case primitiveType::double_ : str = occa::toString(value.double_); break;
      case primitiveType::none :
      default: return "";
    }

    if (type & (primitiveType::uint64_ |
                primitiveType::int64_)) {
      str += 'L';
    }
    return str;
  }

  //---[ Misc Methods ]-----------------
  uint64_t primitive::sizeof_() const {
    switch(type) {
      case primitiveType::bool_   : return sizeof(bool);
      case primitiveType::uint8_  : return sizeof(uint8_t);
      case primitiveType::uint16_ : return sizeof(uint16_t);
      case primitiveType::uint32_ : return sizeof(uint32_t);
      case primitiveType::uint64_ : return sizeof(uint64_t);
      case primitiveType::int8_   : return sizeof(int8_t);
      case primitiveType::int16_  : return sizeof(int16_t);
      case primitiveType::int32_  : return sizeof(int32_t);
      case primitiveType::int64_  : return sizeof(int64_t);
      case primitiveType::float_  : return sizeof(float);
      case primitiveType::double_ : return sizeof(double);
      case primitiveType::ptr     : return sizeof(void*);
      default: return 0;
    }
  }
  //====================================

  //---[ Unary Operators ]--------------
  primitive primitive::not_(const primitive &p) {
    switch(p.type) {
      case primitiveType::bool_   : return primitive(!p.value.bool_);
      case primitiveType::int8_   : return primitive(!p.value.int8_);
      case primitiveType::uint8_  : return primitive(!p.value.uint8_);
      case primitiveType::int16_  : return primitive(!p.value.int16_);
      case primitiveType::uint16_ : return primitive(!p.value.uint16_);
      case primitiveType::int32_  : return primitive(!p.value.int32_);
      case primitiveType::uint32_ : return primitive(!p.value.uint32_);
      case primitiveType::int64_  : return primitive(!p.value.int64_);
      case primitiveType::uint64_ : return primitive(!p.value.uint64_);
      case primitiveType::float_  : OCCA_FORCE_ERROR("Cannot apply operator ! to float type");   break;
      case primitiveType::double_ : OCCA_FORCE_ERROR("Cannot apply operator ! to double type");   break;
      default: ;
    }
    return primitive();
  }

  primitive primitive::positive(const primitive &p) {
    switch(p.type) {
      case primitiveType::bool_   : return primitive(+p.value.bool_);
      case primitiveType::int8_   : return primitive(+p.value.int8_);
      case primitiveType::uint8_  : return primitive(+p.value.uint8_);
      case primitiveType::int16_  : return primitive(+p.value.int16_);
      case primitiveType::uint16_ : return primitive(+p.value.uint16_);
      case primitiveType::int32_  : return primitive(+p.value.int32_);
      case primitiveType::uint32_ : return primitive(+p.value.uint32_);
      case primitiveType::int64_  : return primitive(+p.value.int64_);
      case primitiveType::uint64_ : return primitive(+p.value.uint64_);
      case primitiveType::float_  : return primitive(+p.value.float_);
      case primitiveType::double_ : return primitive(+p.value.double_);
      default: ;
    }
    return primitive();
  }

  primitive primitive::negative(const primitive &p) {
    switch(p.type) {
      case primitiveType::bool_   : return primitive(-p.value.bool_);
      case primitiveType::int8_   : return primitive(-p.value.int8_);
      case primitiveType::uint8_  : return primitive(-p.value.uint8_);
      case primitiveType::int16_  : return primitive(-p.value.int16_);
      case primitiveType::uint16_ : return primitive(-p.value.uint16_);
      case primitiveType::int32_  : return primitive(-p.value.int32_);
      case primitiveType::uint32_ : return primitive(-p.value.uint32_);
      case primitiveType::int64_  : return primitive(-p.value.int64_);
      case primitiveType::uint64_ : return primitive(-p.value.uint64_);
      case primitiveType::float_  : return primitive(-p.value.float_);
      case primitiveType::double_ : return primitive(-p.value.double_);
      default: ;
    }
    return primitive();
  }

  primitive primitive::tilde(const primitive &p) {
    switch(p.type) {
      case primitiveType::bool_   : return primitive(!p.value.bool_);
      case primitiveType::int8_   : return primitive(~p.value.int8_);
      case primitiveType::uint8_  : return primitive(~p.value.uint8_);
      case primitiveType::int16_  : return primitive(~p.value.int16_);
      case primitiveType::uint16_ : return primitive(~p.value.uint16_);
      case primitiveType::int32_  : return primitive(~p.value.int32_);
      case primitiveType::uint32_ : return primitive(~p.value.uint32_);
      case primitiveType::int64_  : return primitive(~p.value.int64_);
      case primitiveType::uint64_ : return primitive(~p.value.uint64_);
      case primitiveType::float_  : OCCA_FORCE_ERROR("Cannot apply operator ~ to float type");   break;
      case primitiveType::double_ : OCCA_FORCE_ERROR("Cannot apply operator ~ to double type");  break;
      default: ;
    }
    return primitive();
  }

  primitive& primitive::leftIncrement(primitive &p) {
    switch(p.type) {
      case primitiveType::bool_   : OCCA_FORCE_ERROR("Cannot apply operator ++ to bool type"); break;
      case primitiveType::int8_   : ++p.value.int8_;    return p;
      case primitiveType::uint8_  : ++p.value.uint8_;   return p;
      case primitiveType::int16_  : ++p.value.int16_;   return p;
      case primitiveType::uint16_ : ++p.value.uint16_;  return p;
      case primitiveType::int32_  : ++p.value.int32_;   return p;
      case primitiveType::uint32_ : ++p.value.uint32_;  return p;
      case primitiveType::int64_  : ++p.value.int64_;   return p;
      case primitiveType::uint64_ : ++p.value.uint64_;  return p;
      case primitiveType::float_  : ++p.value.float_;   return p;
      case primitiveType::double_ : ++p.value.double_;  return p;
      default: ;
    }
    return p;
  }

  primitive& primitive::leftDecrement(primitive &p) {
    switch(p.type) {
      case primitiveType::bool_   : OCCA_FORCE_ERROR("Cannot apply operator -- to bool type"); break;
      case primitiveType::int8_   : --p.value.int8_;    return p;
      case primitiveType::uint8_  : --p.value.uint8_;   return p;
      case primitiveType::int16_  : --p.value.int16_;   return p;
      case primitiveType::uint16_ : --p.value.uint16_;  return p;
      case primitiveType::int32_  : --p.value.int32_;   return p;
      case primitiveType::uint32_ : --p.value.uint32_;  return p;
      case primitiveType::int64_  : --p.value.int64_;   return p;
      case primitiveType::uint64_ : --p.value.uint64_;  return p;
      case primitiveType::float_  : --p.value.float_;   return p;
      case primitiveType::double_ : --p.value.double_;  return p;
      default: ;
    }
    return p;
  }

  primitive primitive::rightIncrement(primitive &p) {
    primitive oldP = p;
    switch(p.type) {
      case primitiveType::bool_   : OCCA_FORCE_ERROR("Cannot apply operator ++ to bool type"); break;
      case primitiveType::int8_   : p.value.int8_++;    return oldP;
      case primitiveType::uint8_  : p.value.uint8_++;   return oldP;
      case primitiveType::int16_  : p.value.int16_++;   return oldP;
      case primitiveType::uint16_ : p.value.uint16_++;  return oldP;
      case primitiveType::int32_  : p.value.int32_++;   return oldP;
      case primitiveType::uint32_ : p.value.uint32_++;  return oldP;
      case primitiveType::int64_  : p.value.int64_++;   return oldP;
      case primitiveType::uint64_ : p.value.uint64_++;  return oldP;
      case primitiveType::float_  : p.value.float_++;   return oldP;
      case primitiveType::double_ : p.value.double_++;  return oldP;
      default: ;
    }
    return oldP;
  }

  primitive primitive::rightDecrement(primitive &p) {
    primitive oldP = p;
    switch(p.type) {
      case primitiveType::bool_   : OCCA_FORCE_ERROR("Cannot apply operator -- to bool type"); break;
      case primitiveType::int8_   : p.value.int8_--;    return oldP;
      case primitiveType::uint8_  : p.value.uint8_--;   return oldP;
      case primitiveType::int16_  : p.value.int16_--;   return oldP;
      case primitiveType::uint16_ : p.value.uint16_--;  return oldP;
      case primitiveType::int32_  : p.value.int32_--;   return oldP;
      case primitiveType::uint32_ : p.value.uint32_--;  return oldP;
      case primitiveType::int64_  : p.value.int64_--;   return oldP;
      case primitiveType::uint64_ : p.value.uint64_--;  return oldP;
      case primitiveType::float_  : p.value.float_--;   return oldP;
      case primitiveType::double_ : p.value.double_--;  return oldP;
      default: ;
    }
    return oldP;
  }
  //====================================


  //---[ Boolean Operators ]------------
  primitive primitive::lessThan(const primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : return primitive(a.to<bool>()     < b.to<bool>());
      case primitiveType::int8_   : return primitive(a.to<int8_t>()   < b.to<int8_t>());
      case primitiveType::uint8_  : return primitive(a.to<uint8_t>()  < b.to<uint8_t>());
      case primitiveType::int16_  : return primitive(a.to<int16_t>()  < b.to<int16_t>());
      case primitiveType::uint16_ : return primitive(a.to<uint16_t>() < b.to<uint16_t>());
      case primitiveType::int32_  : return primitive(a.to<int32_t>()  < b.to<int32_t>());
      case primitiveType::uint32_ : return primitive(a.to<uint32_t>() < b.to<uint32_t>());
      case primitiveType::int64_  : return primitive(a.to<int64_t>()  < b.to<int64_t>());
      case primitiveType::uint64_ : return primitive(a.to<uint64_t>() < b.to<uint64_t>());
      case primitiveType::float_  : return primitive(a.to<float>()    < b.to<float>());
      case primitiveType::double_ : return primitive(a.to<double>()   < b.to<double>());
      default: ;
    }
    return primitive();
  }

  primitive primitive::lessThanEq(const primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : return primitive(a.to<bool>()     <= b.to<bool>());
      case primitiveType::int8_   : return primitive(a.to<int8_t>()   <= b.to<int8_t>());
      case primitiveType::uint8_  : return primitive(a.to<uint8_t>()  <= b.to<uint8_t>());
      case primitiveType::int16_  : return primitive(a.to<int16_t>()  <= b.to<int16_t>());
      case primitiveType::uint16_ : return primitive(a.to<uint16_t>() <= b.to<uint16_t>());
      case primitiveType::int32_  : return primitive(a.to<int32_t>()  <= b.to<int32_t>());
      case primitiveType::uint32_ : return primitive(a.to<uint32_t>() <= b.to<uint32_t>());
      case primitiveType::int64_  : return primitive(a.to<int64_t>()  <= b.to<int64_t>());
      case primitiveType::uint64_ : return primitive(a.to<uint64_t>() <= b.to<uint64_t>());
      case primitiveType::float_  : return primitive(a.to<float>()    <= b.to<float>());
      case primitiveType::double_ : return primitive(a.to<double>()   <= b.to<double>());
      default: ;
    }
    return primitive();
  }

  primitive primitive::equal(const primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : return primitive(a.to<bool>()     == b.to<bool>());
      case primitiveType::int8_   : return primitive(a.to<int8_t>()   == b.to<int8_t>());
      case primitiveType::uint8_  : return primitive(a.to<uint8_t>()  == b.to<uint8_t>());
      case primitiveType::int16_  : return primitive(a.to<int16_t>()  == b.to<int16_t>());
      case primitiveType::uint16_ : return primitive(a.to<uint16_t>() == b.to<uint16_t>());
      case primitiveType::int32_  : return primitive(a.to<int32_t>()  == b.to<int32_t>());
      case primitiveType::uint32_ : return primitive(a.to<uint32_t>() == b.to<uint32_t>());
      case primitiveType::int64_  : return primitive(a.to<int64_t>()  == b.to<int64_t>());
      case primitiveType::uint64_ : return primitive(a.to<uint64_t>() == b.to<uint64_t>());
      case primitiveType::float_  : return primitive(areBitwiseEqual(a.value.float_, b.value.float_));
      case primitiveType::double_ : return primitive(areBitwiseEqual(a.value.double_, b.value.double_));
      default: ;
    }
    return primitive();
  }

  primitive primitive::compare(const primitive &a, const primitive &b) {
    if (lessThan(a, b)) {
      return -1;
    }
    return greaterThan(a, b) ? 1 : 0;
  }

  primitive primitive::notEqual(const primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : return primitive(a.to<bool>()     != b.to<bool>());
      case primitiveType::int8_   : return primitive(a.to<int8_t>()   != b.to<int8_t>());
      case primitiveType::uint8_  : return primitive(a.to<uint8_t>()  != b.to<uint8_t>());
      case primitiveType::int16_  : return primitive(a.to<int16_t>()  != b.to<int16_t>());
      case primitiveType::uint16_ : return primitive(a.to<uint16_t>() != b.to<uint16_t>());
      case primitiveType::int32_  : return primitive(a.to<int32_t>()  != b.to<int32_t>());
      case primitiveType::uint32_ : return primitive(a.to<uint32_t>() != b.to<uint32_t>());
      case primitiveType::int64_  : return primitive(a.to<int64_t>()  != b.to<int64_t>());
      case primitiveType::uint64_ : return primitive(a.to<uint64_t>() != b.to<uint64_t>());
      case primitiveType::float_  : return primitive(!areBitwiseEqual(a.value.float_, b.value.float_));
      case primitiveType::double_ : return primitive(!areBitwiseEqual(a.value.double_, b.value.double_));
      default: ;
    }
    return primitive();
  }

  primitive primitive::greaterThanEq(const primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : return primitive(a.to<bool>()     >= b.to<bool>());
      case primitiveType::int8_   : return primitive(a.to<int8_t>()   >= b.to<int8_t>());
      case primitiveType::uint8_  : return primitive(a.to<uint8_t>()  >= b.to<uint8_t>());
      case primitiveType::int16_  : return primitive(a.to<int16_t>()  >= b.to<int16_t>());
      case primitiveType::uint16_ : return primitive(a.to<uint16_t>() >= b.to<uint16_t>());
      case primitiveType::int32_  : return primitive(a.to<int32_t>()  >= b.to<int32_t>());
      case primitiveType::uint32_ : return primitive(a.to<uint32_t>() >= b.to<uint32_t>());
      case primitiveType::int64_  : return primitive(a.to<int64_t>()  >= b.to<int64_t>());
      case primitiveType::uint64_ : return primitive(a.to<uint64_t>() >= b.to<uint64_t>());
      case primitiveType::float_  : return primitive(a.to<float>()    >= b.to<float>());
      case primitiveType::double_ : return primitive(a.to<double>()   >= b.to<double>());
      default: ;
    }
    return primitive();
  }

  primitive primitive::greaterThan(const primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : return primitive(a.to<bool>()     > b.to<bool>());
      case primitiveType::int8_   : return primitive(a.to<int8_t>()   > b.to<int8_t>());
      case primitiveType::uint8_  : return primitive(a.to<uint8_t>()  > b.to<uint8_t>());
      case primitiveType::int16_  : return primitive(a.to<int16_t>()  > b.to<int16_t>());
      case primitiveType::uint16_ : return primitive(a.to<uint16_t>() > b.to<uint16_t>());
      case primitiveType::int32_  : return primitive(a.to<int32_t>()  > b.to<int32_t>());
      case primitiveType::uint32_ : return primitive(a.to<uint32_t>() > b.to<uint32_t>());
      case primitiveType::int64_  : return primitive(a.to<int64_t>()  > b.to<int64_t>());
      case primitiveType::uint64_ : return primitive(a.to<uint64_t>() > b.to<uint64_t>());
      case primitiveType::float_  : return primitive(a.to<float>()    > b.to<float>());
      case primitiveType::double_ : return primitive(a.to<double>()   > b.to<double>());
      default: ;
    }
    return primitive();
  }

  primitive primitive::and_(const primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : return primitive(a.to<bool>()     && b.to<bool>());
      case primitiveType::int8_   : return primitive(a.to<int8_t>()   && b.to<int8_t>());
      case primitiveType::uint8_  : return primitive(a.to<uint8_t>()  && b.to<uint8_t>());
      case primitiveType::int16_  : return primitive(a.to<int16_t>()  && b.to<int16_t>());
      case primitiveType::uint16_ : return primitive(a.to<uint16_t>() && b.to<uint16_t>());
      case primitiveType::int32_  : return primitive(a.to<int32_t>()  && b.to<int32_t>());
      case primitiveType::uint32_ : return primitive(a.to<uint32_t>() && b.to<uint32_t>());
      case primitiveType::int64_  : return primitive(a.to<int64_t>()  && b.to<int64_t>());
      case primitiveType::uint64_ : return primitive(a.to<uint64_t>() && b.to<uint64_t>());
      case primitiveType::float_  : OCCA_FORCE_ERROR("Cannot apply operator && to float type");   break;
      case primitiveType::double_ : OCCA_FORCE_ERROR("Cannot apply operator && to double type");   break;
      default: ;
    }
    return primitive();
  }

  primitive primitive::or_(const primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : return primitive(a.to<bool>()     || b.to<bool>());
      case primitiveType::int8_   : return primitive(a.to<int8_t>()   || b.to<int8_t>());
      case primitiveType::uint8_  : return primitive(a.to<uint8_t>()  || b.to<uint8_t>());
      case primitiveType::int16_  : return primitive(a.to<int16_t>()  || b.to<int16_t>());
      case primitiveType::uint16_ : return primitive(a.to<uint16_t>() || b.to<uint16_t>());
      case primitiveType::int32_  : return primitive(a.to<int32_t>()  || b.to<int32_t>());
      case primitiveType::uint32_ : return primitive(a.to<uint32_t>() || b.to<uint32_t>());
      case primitiveType::int64_  : return primitive(a.to<int64_t>()  || b.to<int64_t>());
      case primitiveType::uint64_ : return primitive(a.to<uint64_t>() || b.to<uint64_t>());
      case primitiveType::float_  : OCCA_FORCE_ERROR("Cannot apply operator || to float type");   break;
      case primitiveType::double_ : OCCA_FORCE_ERROR("Cannot apply operator || to double type");   break;
      default: ;
    }
    return primitive();
  }
  //====================================


  //---[ Binary Operators ]-------------
  primitive primitive::mult(const primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : return primitive(a.to<bool>()     * b.to<bool>());
      case primitiveType::int8_   : return primitive(a.to<int8_t>()   * b.to<int8_t>());
      case primitiveType::uint8_  : return primitive(a.to<uint8_t>()  * b.to<uint8_t>());
      case primitiveType::int16_  : return primitive(a.to<int16_t>()  * b.to<int16_t>());
      case primitiveType::uint16_ : return primitive(a.to<uint16_t>() * b.to<uint16_t>());
      case primitiveType::int32_  : return primitive(a.to<int32_t>()  * b.to<int32_t>());
      case primitiveType::uint32_ : return primitive(a.to<uint32_t>() * b.to<uint32_t>());
      case primitiveType::int64_  : return primitive(a.to<int64_t>()  * b.to<int64_t>());
      case primitiveType::uint64_ : return primitive(a.to<uint64_t>() * b.to<uint64_t>());
      case primitiveType::float_  : return primitive(a.to<float>()    * b.to<float>());
      case primitiveType::double_ : return primitive(a.to<double>()   * b.to<double>());
      default: ;
    }
    return primitive();
  }

  primitive primitive::add(const primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : return primitive(a.to<bool>()     + b.to<bool>());
      case primitiveType::int8_   : return primitive(a.to<int8_t>()   + b.to<int8_t>());
      case primitiveType::uint8_  : return primitive(a.to<uint8_t>()  + b.to<uint8_t>());
      case primitiveType::int16_  : return primitive(a.to<int16_t>()  + b.to<int16_t>());
      case primitiveType::uint16_ : return primitive(a.to<uint16_t>() + b.to<uint16_t>());
      case primitiveType::int32_  : return primitive(a.to<int32_t>()  + b.to<int32_t>());
      case primitiveType::uint32_ : return primitive(a.to<uint32_t>() + b.to<uint32_t>());
      case primitiveType::int64_  : return primitive(a.to<int64_t>()  + b.to<int64_t>());
      case primitiveType::uint64_ : return primitive(a.to<uint64_t>() + b.to<uint64_t>());
      case primitiveType::float_  : return primitive(a.to<float>()    + b.to<float>());
      case primitiveType::double_ : return primitive(a.to<double>()   + b.to<double>());
      default: ;
    }
    return primitive();
  }

  primitive primitive::sub(const primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : return primitive(a.to<bool>()     - b.to<bool>());
      case primitiveType::int8_   : return primitive(a.to<int8_t>()   - b.to<int8_t>());
      case primitiveType::uint8_  : return primitive(a.to<uint8_t>()  - b.to<uint8_t>());
      case primitiveType::int16_  : return primitive(a.to<int16_t>()  - b.to<int16_t>());
      case primitiveType::uint16_ : return primitive(a.to<uint16_t>() - b.to<uint16_t>());
      case primitiveType::int32_  : return primitive(a.to<int32_t>()  - b.to<int32_t>());
      case primitiveType::uint32_ : return primitive(a.to<uint32_t>() - b.to<uint32_t>());
      case primitiveType::int64_  : return primitive(a.to<int64_t>()  - b.to<int64_t>());
      case primitiveType::uint64_ : return primitive(a.to<uint64_t>() - b.to<uint64_t>());
      case primitiveType::float_  : return primitive(a.to<float>()    - b.to<float>());
      case primitiveType::double_ : return primitive(a.to<double>()   - b.to<double>());
      default: ;
    }
    return primitive();
  }

  primitive primitive::div(const primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : return primitive(a.to<bool>()     / b.to<bool>());
      case primitiveType::int8_   : return primitive(a.to<int8_t>()   / b.to<int8_t>());
      case primitiveType::uint8_  : return primitive(a.to<uint8_t>()  / b.to<uint8_t>());
      case primitiveType::int16_  : return primitive(a.to<int16_t>()  / b.to<int16_t>());
      case primitiveType::uint16_ : return primitive(a.to<uint16_t>() / b.to<uint16_t>());
      case primitiveType::int32_  : return primitive(a.to<int32_t>()  / b.to<int32_t>());
      case primitiveType::uint32_ : return primitive(a.to<uint32_t>() / b.to<uint32_t>());
      case primitiveType::int64_  : return primitive(a.to<int64_t>()  / b.to<int64_t>());
      case primitiveType::uint64_ : return primitive(a.to<uint64_t>() / b.to<uint64_t>());
      case primitiveType::float_  : return primitive(a.to<float>()    / b.to<float>());
      case primitiveType::double_ : return primitive(a.to<double>()   / b.to<double>());
      default: ;
    }
    return primitive();
  }

  primitive primitive::mod(const primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : return primitive(a.to<bool>()     % b.to<bool>());
      case primitiveType::int8_   : return primitive(a.to<int8_t>()   % b.to<int8_t>());
      case primitiveType::uint8_  : return primitive(a.to<uint8_t>()  % b.to<uint8_t>());
      case primitiveType::int16_  : return primitive(a.to<int16_t>()  % b.to<int16_t>());
      case primitiveType::uint16_ : return primitive(a.to<uint16_t>() % b.to<uint16_t>());
      case primitiveType::int32_  : return primitive(a.to<int32_t>()  % b.to<int32_t>());
      case primitiveType::uint32_ : return primitive(a.to<uint32_t>() % b.to<uint32_t>());
      case primitiveType::int64_  : return primitive(a.to<int64_t>()  % b.to<int64_t>());
      case primitiveType::uint64_ : return primitive(a.to<uint64_t>() % b.to<uint64_t>());
      case primitiveType::float_  : OCCA_FORCE_ERROR("Cannot apply operator % to float type"); break;
      case primitiveType::double_ : OCCA_FORCE_ERROR("Cannot apply operator % to double type"); break;
      default: ;
    }
    return primitive();
  }

  primitive primitive::bitAnd(const primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : OCCA_FORCE_ERROR("Cannot apply operator & to bool type");   break;
      case primitiveType::int8_   : return primitive(a.to<int8_t>()   & b.to<int8_t>());
      case primitiveType::uint8_  : return primitive(a.to<uint8_t>()  & b.to<uint8_t>());
      case primitiveType::int16_  : return primitive(a.to<int16_t>()  & b.to<int16_t>());
      case primitiveType::uint16_ : return primitive(a.to<uint16_t>() & b.to<uint16_t>());
      case primitiveType::int32_  : return primitive(a.to<int32_t>()  & b.to<int32_t>());
      case primitiveType::uint32_ : return primitive(a.to<uint32_t>() & b.to<uint32_t>());
      case primitiveType::int64_  : return primitive(a.to<int64_t>()  & b.to<int64_t>());
      case primitiveType::uint64_ : return primitive(a.to<uint64_t>() & b.to<uint64_t>());
      case primitiveType::float_  : OCCA_FORCE_ERROR("Cannot apply operator & to float type");   break;
      case primitiveType::double_ : OCCA_FORCE_ERROR("Cannot apply operator & to double type");  break;
      default: ;
    }
    return primitive();
  }

  primitive primitive::bitOr(const primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : OCCA_FORCE_ERROR("Cannot apply operator | to bool type");   break;
      case primitiveType::int8_   : return primitive(a.to<int8_t>()   | b.to<int8_t>());
      case primitiveType::uint8_  : return primitive(a.to<uint8_t>()  | b.to<uint8_t>());
      case primitiveType::int16_  : return primitive(a.to<int16_t>()  | b.to<int16_t>());
      case primitiveType::uint16_ : return primitive(a.to<uint16_t>() | b.to<uint16_t>());
      case primitiveType::int32_  : return primitive(a.to<int32_t>()  | b.to<int32_t>());
      case primitiveType::uint32_ : return primitive(a.to<uint32_t>() | b.to<uint32_t>());
      case primitiveType::int64_  : return primitive(a.to<int64_t>()  | b.to<int64_t>());
      case primitiveType::uint64_ : return primitive(a.to<uint64_t>() | b.to<uint64_t>());
      case primitiveType::float_  : OCCA_FORCE_ERROR("Cannot apply operator | to float type");   break;
      case primitiveType::double_ : OCCA_FORCE_ERROR("Cannot apply operator | to double type");  break;
      default: ;
    }
    return primitive();
  }

  primitive primitive::xor_(const primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : OCCA_FORCE_ERROR("Cannot apply operator ^ to bool type");   break;
      case primitiveType::int8_   : return primitive(a.to<int8_t>()   ^ b.to<int8_t>());
      case primitiveType::uint8_  : return primitive(a.to<uint8_t>()  ^ b.to<uint8_t>());
      case primitiveType::int16_  : return primitive(a.to<int16_t>()  ^ b.to<int16_t>());
      case primitiveType::uint16_ : return primitive(a.to<uint16_t>() ^ b.to<uint16_t>());
      case primitiveType::int32_  : return primitive(a.to<int32_t>()  ^ b.to<int32_t>());
      case primitiveType::uint32_ : return primitive(a.to<uint32_t>() ^ b.to<uint32_t>());
      case primitiveType::int64_  : return primitive(a.to<int64_t>()  ^ b.to<int64_t>());
      case primitiveType::uint64_ : return primitive(a.to<uint64_t>() ^ b.to<uint64_t>());
      case primitiveType::float_  : OCCA_FORCE_ERROR("Cannot apply operator ^ to float type");   break;
      case primitiveType::double_ : OCCA_FORCE_ERROR("Cannot apply operator ^ to double type");  break;
      default: ;
    }
    return primitive();
  }

  primitive primitive::rightShift(const primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : return primitive(a.to<bool>()     >> b.to<bool>());
      case primitiveType::int8_   : return primitive(a.to<int8_t>()   >> b.to<int8_t>());
      case primitiveType::uint8_  : return primitive(a.to<uint8_t>()  >> b.to<uint8_t>());
      case primitiveType::int16_  : return primitive(a.to<int16_t>()  >> b.to<int16_t>());
      case primitiveType::uint16_ : return primitive(a.to<uint16_t>() >> b.to<uint16_t>());
      case primitiveType::int32_  : return primitive(a.to<int32_t>()  >> b.to<int32_t>());
      case primitiveType::uint32_ : return primitive(a.to<uint32_t>() >> b.to<uint32_t>());
      case primitiveType::int64_  : return primitive(a.to<int64_t>()  >> b.to<int64_t>());
      case primitiveType::uint64_ : return primitive(a.to<uint64_t>() >> b.to<uint64_t>());
      case primitiveType::float_  : OCCA_FORCE_ERROR("Cannot apply operator >> to float type");   break;
      case primitiveType::double_ : OCCA_FORCE_ERROR("Cannot apply operator >> to double type");  break;
      default: ;
    }
    return primitive();
  }

  primitive primitive::leftShift(const primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : return primitive(a.to<bool>()     << b.to<bool>());
      case primitiveType::int8_   : return primitive(a.to<int8_t>()   << b.to<int8_t>());
      case primitiveType::uint8_  : return primitive(a.to<uint8_t>()  << b.to<uint8_t>());
      case primitiveType::int16_  : return primitive(a.to<int16_t>()  << b.to<int16_t>());
      case primitiveType::uint16_ : return primitive(a.to<uint16_t>() << b.to<uint16_t>());
      case primitiveType::int32_  : return primitive(a.to<int32_t>()  << b.to<int32_t>());
      case primitiveType::uint32_ : return primitive(a.to<uint32_t>() << b.to<uint32_t>());
      case primitiveType::int64_  : return primitive(a.to<int64_t>()  << b.to<int64_t>());
      case primitiveType::uint64_ : return primitive(a.to<uint64_t>() << b.to<uint64_t>());
      case primitiveType::float_  : OCCA_FORCE_ERROR("Cannot apply operator << to float type");   break;
      case primitiveType::double_ : OCCA_FORCE_ERROR("Cannot apply operator << to double type");  break;
      default: ;
    }
    return primitive();
  }
  //====================================


  //---[ Assignment Operators ]---------
  primitive& primitive::assign(primitive &a, const primitive &b) {
    a = b;
    return a;
  }

  primitive& primitive::multEq(primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : a = (a.to<bool>()     * b.to<bool>());     break;
      case primitiveType::int8_   : a = (a.to<int8_t>()   * b.to<int8_t>());   break;
      case primitiveType::uint8_  : a = (a.to<uint8_t>()  * b.to<uint8_t>());  break;
      case primitiveType::int16_  : a = (a.to<int16_t>()  * b.to<int16_t>());  break;
      case primitiveType::uint16_ : a = (a.to<uint16_t>() * b.to<uint16_t>()); break;
      case primitiveType::int32_  : a = (a.to<int32_t>()  * b.to<int32_t>());  break;
      case primitiveType::uint32_ : a = (a.to<uint32_t>() * b.to<uint32_t>()); break;
      case primitiveType::int64_  : a = (a.to<int64_t>()  * b.to<int64_t>());  break;
      case primitiveType::uint64_ : a = (a.to<uint64_t>() * b.to<uint64_t>()); break;
      case primitiveType::float_  : a = (a.to<float>()    * b.to<float>());    break;
      case primitiveType::double_ : a = (a.to<double>()   * b.to<double>());   break;
      default: ;
    }
    return a;
  }

  primitive& primitive::addEq(primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : a = (a.to<bool>()     + b.to<bool>());     break;
      case primitiveType::int8_   : a = (a.to<int8_t>()   + b.to<int8_t>());   break;
      case primitiveType::uint8_  : a = (a.to<uint8_t>()  + b.to<uint8_t>());  break;
      case primitiveType::int16_  : a = (a.to<int16_t>()  + b.to<int16_t>());  break;
      case primitiveType::uint16_ : a = (a.to<uint16_t>() + b.to<uint16_t>()); break;
      case primitiveType::int32_  : a = (a.to<int32_t>()  + b.to<int32_t>());  break;
      case primitiveType::uint32_ : a = (a.to<uint32_t>() + b.to<uint32_t>()); break;
      case primitiveType::int64_  : a = (a.to<int64_t>()  + b.to<int64_t>());  break;
      case primitiveType::uint64_ : a = (a.to<uint64_t>() + b.to<uint64_t>()); break;
      case primitiveType::float_  : a = (a.to<float>()    + b.to<float>());    break;
      case primitiveType::double_ : a = (a.to<double>()   + b.to<double>());   break;
      default: ;
    }
    return a;
  }

  primitive& primitive::subEq(primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : a = (a.to<bool>()     - b.to<bool>());     break;
      case primitiveType::int8_   : a = (a.to<int8_t>()   - b.to<int8_t>());   break;
      case primitiveType::uint8_  : a = (a.to<uint8_t>()  - b.to<uint8_t>());  break;
      case primitiveType::int16_  : a = (a.to<int16_t>()  - b.to<int16_t>());  break;
      case primitiveType::uint16_ : a = (a.to<uint16_t>() - b.to<uint16_t>()); break;
      case primitiveType::int32_  : a = (a.to<int32_t>()  - b.to<int32_t>());  break;
      case primitiveType::uint32_ : a = (a.to<uint32_t>() - b.to<uint32_t>()); break;
      case primitiveType::int64_  : a = (a.to<int64_t>()  - b.to<int64_t>());  break;
      case primitiveType::uint64_ : a = (a.to<uint64_t>() - b.to<uint64_t>()); break;
      case primitiveType::float_  : a = (a.to<float>()    - b.to<float>());    break;
      case primitiveType::double_ : a = (a.to<double>()   - b.to<double>());   break;
      default: ;
    }
    return a;
  }

  primitive& primitive::divEq(primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : a = (a.to<bool>()     / b.to<bool>());     break;
      case primitiveType::int8_   : a = (a.to<int8_t>()   / b.to<int8_t>());   break;
      case primitiveType::uint8_  : a = (a.to<uint8_t>()  / b.to<uint8_t>());  break;
      case primitiveType::int16_  : a = (a.to<int16_t>()  / b.to<int16_t>());  break;
      case primitiveType::uint16_ : a = (a.to<uint16_t>() / b.to<uint16_t>()); break;
      case primitiveType::int32_  : a = (a.to<int32_t>()  / b.to<int32_t>());  break;
      case primitiveType::uint32_ : a = (a.to<uint32_t>() / b.to<uint32_t>()); break;
      case primitiveType::int64_  : a = (a.to<int64_t>()  / b.to<int64_t>());  break;
      case primitiveType::uint64_ : a = (a.to<uint64_t>() / b.to<uint64_t>()); break;
      case primitiveType::float_  : a = (a.to<float>()    / b.to<float>());    break;
      case primitiveType::double_ : a = (a.to<double>()   / b.to<double>());   break;
      default: ;
    }
    return a;
  }

  primitive& primitive::modEq(primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : a = (a.to<bool>()     % b.to<bool>());     break;
      case primitiveType::int8_   : a = (a.to<int8_t>()   % b.to<int8_t>());   break;
      case primitiveType::uint8_  : a = (a.to<uint8_t>()  % b.to<uint8_t>());  break;
      case primitiveType::int16_  : a = (a.to<int16_t>()  % b.to<int16_t>());  break;
      case primitiveType::uint16_ : a = (a.to<uint16_t>() % b.to<uint16_t>()); break;
      case primitiveType::int32_  : a = (a.to<int32_t>()  % b.to<int32_t>());  break;
      case primitiveType::uint32_ : a = (a.to<uint32_t>() % b.to<uint32_t>()); break;
      case primitiveType::int64_  : a = (a.to<int64_t>()  % b.to<int64_t>());  break;
      case primitiveType::uint64_ : a = (a.to<uint64_t>() % b.to<uint64_t>()); break;
      case primitiveType::float_  : OCCA_FORCE_ERROR("Cannot apply operator % to float type"); break;
      case primitiveType::double_ : OCCA_FORCE_ERROR("Cannot apply operator % to double type"); break;
      default: ;
    }
    return a;
  }

  primitive& primitive::bitAndEq(primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : OCCA_FORCE_ERROR("Cannot apply operator &= to bool type");  break;
      case primitiveType::int8_   : a = (a.to<int8_t>()   & b.to<int8_t>());   break;
      case primitiveType::uint8_  : a = (a.to<uint8_t>()  & b.to<uint8_t>());  break;
      case primitiveType::int16_  : a = (a.to<int16_t>()  & b.to<int16_t>());  break;
      case primitiveType::uint16_ : a = (a.to<uint16_t>() & b.to<uint16_t>()); break;
      case primitiveType::int32_  : a = (a.to<int32_t>()  & b.to<int32_t>());  break;
      case primitiveType::uint32_ : a = (a.to<uint32_t>() & b.to<uint32_t>()); break;
      case primitiveType::int64_  : a = (a.to<int64_t>()  & b.to<int64_t>());  break;
      case primitiveType::uint64_ : a = (a.to<uint64_t>() & b.to<uint64_t>()); break;
      case primitiveType::float_  : OCCA_FORCE_ERROR("Cannot apply operator &= to float type");  break;
      case primitiveType::double_ : OCCA_FORCE_ERROR("Cannot apply operator &= to double type"); break;
      default: ;
    }
    return a;
  }

  primitive& primitive::bitOrEq(primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : OCCA_FORCE_ERROR("Cannot apply operator |= to bool type");  break;
      case primitiveType::int8_   : a = (a.to<int8_t>()   | b.to<int8_t>());   break;
      case primitiveType::uint8_  : a = (a.to<uint8_t>()  | b.to<uint8_t>());  break;
      case primitiveType::int16_  : a = (a.to<int16_t>()  | b.to<int16_t>());  break;
      case primitiveType::uint16_ : a = (a.to<uint16_t>() | b.to<uint16_t>()); break;
      case primitiveType::int32_  : a = (a.to<int32_t>()  | b.to<int32_t>());  break;
      case primitiveType::uint32_ : a = (a.to<uint32_t>() | b.to<uint32_t>()); break;
      case primitiveType::int64_  : a = (a.to<int64_t>()  | b.to<int64_t>());  break;
      case primitiveType::uint64_ : a = (a.to<uint64_t>() | b.to<uint64_t>()); break;
      case primitiveType::float_  : OCCA_FORCE_ERROR("Cannot apply operator |= to float type");  break;
      case primitiveType::double_ : OCCA_FORCE_ERROR("Cannot apply operator |= to double type"); break;
      default: ;
    }
    return a;
  }

  primitive& primitive::xorEq(primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : OCCA_FORCE_ERROR("Cannot apply operator ^= to bool type");  break;
      case primitiveType::int8_   : a = (a.to<int8_t>()   ^ b.to<int8_t>());   break;
      case primitiveType::uint8_  : a = (a.to<uint8_t>()  ^ b.to<uint8_t>());  break;
      case primitiveType::int16_  : a = (a.to<int16_t>()  ^ b.to<int16_t>());  break;
      case primitiveType::uint16_ : a = (a.to<uint16_t>() ^ b.to<uint16_t>()); break;
      case primitiveType::int32_  : a = (a.to<int32_t>()  ^ b.to<int32_t>());  break;
      case primitiveType::uint32_ : a = (a.to<uint32_t>() ^ b.to<uint32_t>()); break;
      case primitiveType::int64_  : a = (a.to<int64_t>()  ^ b.to<int64_t>());  break;
      case primitiveType::uint64_ : a = (a.to<uint64_t>() ^ b.to<uint64_t>()); break;
      case primitiveType::float_  : OCCA_FORCE_ERROR("Cannot apply operator ^= to float type");  break;
      case primitiveType::double_ : OCCA_FORCE_ERROR("Cannot apply operator ^= to double type"); break;
      default: ;
    }
    return a;
  }

  primitive& primitive::rightShiftEq(primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : a = (a.to<bool>()     >> b.to<bool>());     break;
      case primitiveType::int8_   : a = (a.to<int8_t>()   >> b.to<int8_t>());   break;
      case primitiveType::uint8_  : a = (a.to<uint8_t>()  >> b.to<uint8_t>());  break;
      case primitiveType::int16_  : a = (a.to<int16_t>()  >> b.to<int16_t>());  break;
      case primitiveType::uint16_ : a = (a.to<uint16_t>() >> b.to<uint16_t>()); break;
      case primitiveType::int32_  : a = (a.to<int32_t>()  >> b.to<int32_t>());  break;
      case primitiveType::uint32_ : a = (a.to<uint32_t>() >> b.to<uint32_t>()); break;
      case primitiveType::int64_  : a = (a.to<int64_t>()  >> b.to<int64_t>());  break;
      case primitiveType::uint64_ : a = (a.to<uint64_t>() >> b.to<uint64_t>()); break;
      case primitiveType::float_  : OCCA_FORCE_ERROR("Cannot apply operator >> to float type");  break;
      case primitiveType::double_ : OCCA_FORCE_ERROR("Cannot apply operator >> to double type"); break;
      default: ;
    }
    return a;
  }

  primitive& primitive::leftShiftEq(primitive &a, const primitive &b) {
    const int retType = (a.type > b.type) ? a.type : b.type;
    switch(retType) {
      case primitiveType::bool_   : a = (a.to<bool>()     << b.to<bool>());     break;
      case primitiveType::int8_   : a = (a.to<int8_t>()   << b.to<int8_t>());   break;
      case primitiveType::uint8_  : a = (a.to<uint8_t>()  << b.to<uint8_t>());  break;
      case primitiveType::int16_  : a = (a.to<int16_t>()  << b.to<int16_t>());  break;
      case primitiveType::uint16_ : a = (a.to<uint16_t>() << b.to<uint16_t>()); break;
      case primitiveType::int32_  : a = (a.to<int32_t>()  << b.to<int32_t>());  break;
      case primitiveType::uint32_ : a = (a.to<uint32_t>() << b.to<uint32_t>()); break;
      case primitiveType::int64_  : a = (a.to<int64_t>()  << b.to<int64_t>());  break;
      case primitiveType::uint64_ : a = (a.to<uint64_t>() << b.to<uint64_t>()); break;
      case primitiveType::float_  : OCCA_FORCE_ERROR("Cannot apply operator << to float type");  break;
      case primitiveType::double_ : OCCA_FORCE_ERROR("Cannot apply operator << to double type"); break;
      default: ;
    }
    return a;
  }
  //====================================

  io::output& operator << (io::output &out, const primitive &p) {
    out << p.toString();
    return out;
  }
}
