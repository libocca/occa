#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <cstdlib>

#include "occa/defines.hpp"

static const char wordDelimiter1[]  = " \t\r\n\v\f!\"#%&'()*+,-./:;<=>?[]^{|}~@\0";

bool charIsIn(const char c, const char *delimiters) {
  while ((*delimiters) != '\0')
    if (c == *(delimiters++))
      return true;

  return false;
}

char upChar(const char c) {
  if (('a' <= c) && (c <= 'z'))
    return ((c + 'A') - 'a');

  return c;
}

namespace occa {
  int atoi(const std::string &str) {
    return ::atoi(str.c_str());
  }

  double atof(const std::string &str) {
    return ::atof(str.c_str());
  }

  double atod(const char *c) {
    double ret;
    sscanf(c, "%lf", &ret);
    return ret;
  }

  double atod(const std::string &str) {
    return atod(str.c_str());
  }

  template <class TM>
  std::string toString(const TM &t) {
    std::stringstream ss;
    ss << std::setprecision(100) << t;
    return ss.str();
  }
}

enum type_t {
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

  primitive() :
    type(none) {
    value.ptr = NULL;
  }

  primitive(const primitive &p) :
    type(p.type) {
    value.ptr = p.value.ptr;
  }

  primitive(const char *c);
  primitive(const std::string &s);

  primitive(const uint8_t value_) {
    type = uint8_;
    value.uint8_ = (uint8_t) value_;
  }

  primitive(const uint16_t value_) {
    type = uint16_;
    value.uint16_ = (uint16_t) value_;
  }

  primitive(const uint32_t value_) {
    type = uint32_;
    value.uint32_ = (uint32_t) value_;
  }

  primitive(const uint64_t value_) {
    type = uint64_;
    value.uint64_ = (uint64_t) value_;
  }

  primitive(const int8_t value_) {
    type = int8_;
    value.int8_ = (int8_t) value_;
  }

  primitive(const int16_t value_) {
    type = int16_;
    value.int16_ = (int16_t) value_;
  }

  primitive(const int32_t value_) {
    type = int32_;
    value.int32_ = (int32_t) value_;
  }

  primitive(const int64_t value_) {
    type = int64_;
    value.int64_ = (int64_t) value_;
  }

  primitive(const float value_) {
    type = float_;
    value.float_ = value_;
  }

  primitive(const double value_) {
    type = double_;
    value.double_ = value_;
  }

  primitive(void *value_) {
    type = ptr;
    value.ptr = (char*) value_;
  }

  static primitive load(const char *&c);
  static primitive load(const std::string &s);

  static primitive loadBinary(const char *&c, const bool isNegative = false);
  static primitive loadHex(const char *&c, const bool isNegative = false);

  primitive& operator = (const uint8_t value_) {
    type = uint8_;
    value.uint8_ = (uint8_t) value_;
    return *this;
  }

  primitive& operator = (const uint16_t value_) {
    type = uint16_;
    value.uint16_ = (uint16_t) value_;
    return *this;
  }

  primitive& operator = (const uint32_t value_) {
    type = uint32_;
    value.uint32_ = (uint32_t) value_;
    return *this;
  }

  primitive& operator = (const uint64_t value_) {
    type = uint64_;
    value.uint64_ = (uint64_t) value_;
    return *this;
  }

  primitive& operator = (const int8_t value_) {
    type = int8_;
    value.int8_ = (int8_t) value_;
    return *this;
  }

  primitive& operator = (const int16_t value_) {
    type = int16_;
    value.int16_ = (int16_t) value_;
    return *this;
  }

  primitive& operator = (const int32_t value_) {
    type = int32_;
    value.int32_ = (int32_t) value_;
    return *this;
  }

  primitive& operator = (const int64_t value_) {
    type = int64_;
    value.int64_ = (int64_t) value_;
    return *this;
  }

  primitive& operator = (const float value_) {
    type = float_;
    value.float_ = value_;
    return *this;
  }

  primitive& operator = (const double value_) {
    type = double_;
    value.double_ = value_;
    return *this;
  }

  primitive& operator = (void *value_) {
    type = ptr;
    value.ptr = (char*) value_;
    return *this;
  }

  operator uint8_t () {
    return to<uint8_t>();
  }

  operator uint16_t () {
    return to<uint16_t>();
  }

  operator uint32_t () {
    return to<uint32_t>();
  }

  operator uint64_t () {
    return to<uint64_t>();
  }

  operator int8_t () {
    return to<int8_t>();
  }

  operator int16_t () {
    return to<int16_t>();
  }

  operator int32_t () {
    return to<int32_t>();
  }

  operator int64_t () {
    return to<int64_t>();
  }

  operator float () {
    return to<float>();
  }

  operator double () {
    return to<double>();
  }

  template <class TM>
  TM to() const {
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

  bool isSigned() const {
    return type & ::isSigned;
  }

  bool isUnsigned() const {
    return type & ::isUnsigned;
  }

  operator std::string () const;

  //   ---[ Unary Operators ]-----------------
  primitive  operator ! ();
  primitive  operator + ();
  primitive  operator - ();
  primitive  operator ~ ();
  primitive& operator ++ ();
  primitive& operator -- ();

  primitive operator ++ (int);
  primitive operator -- (int);
  //    ======================================


  //   ---[ Boolean Operators ]---------------
  primitive operator < (const primitive &p);
  primitive operator <= (const primitive &p);
  primitive operator == (const primitive &p);
  primitive operator != (const primitive &p);
  primitive operator >= (const primitive &p);
  primitive operator > (const primitive &p);

  primitive operator && (const primitive &p);
  primitive operator || (const primitive &p);
  //    ======================================


  //   ---[ Binary Operators ]----------------
  primitive operator * (const primitive &p);
  primitive operator + (const primitive &p);
  primitive operator - (const primitive &p);
  primitive operator / (const primitive &p);
  primitive operator % (const primitive &p);

  primitive operator & (const primitive &p);
  primitive operator | (const primitive &p);
  primitive operator ^ (const primitive &p);

  primitive operator >> (const primitive &p);
  primitive operator << (const primitive &p);
  //   =======================================


  //   ---[ Assignment Operators ]--------------
  primitive operator *= (const primitive &p);
  primitive operator += (const primitive &p);
  primitive operator -= (const primitive &p);
  primitive operator /= (const primitive &p);
  primitive operator %= (const primitive &p);

  primitive operator &= (const primitive &p);
  primitive operator |= (const primitive &p);
  primitive operator ^= (const primitive &p);

  primitive operator >>= (const primitive &p);
  primitive operator <<= (const primitive &p);
  //   =======================================

  friend std::ostream& operator << (std::ostream &out, const primitive &p);
};

primitive::primitive(const char *c) {
  *this = load(c);
}

primitive::primitive(const std::string &s) {
  const char *c = s.c_str();
  *this = load(c);
}

primitive primitive::load(const char *&c) {
  bool unsigned_ = false;
  bool negative  = false;
  bool decimal   = false;
  bool float_    = false;
  int longs      = 0;
  int digits     = 0;

  const char *c0 = c;
  primitive p;

  if ((strcmp(c, "true")  == 0) ||
      (strcmp(c, "false") == 0)) {
    p = (uint8_t) (strcmp(c, "true") == 0);
    c += (p.value.uint8_ ? 4 : 5);
    return p;
  }

  if ((*c == '+') || (*c == '-')) {
    negative = (*c == '-');
    ++c;
  }

  if (*c == '0') {
    ++digits;
    ++c;

    const char C = upChar(*c);

    if (C == 'B') {
      return primitive::loadBinary(++c, negative);
    } else if (C == 'X') {
      return primitive::loadHex(++c, negative);
    } else {
      --c;
    }
  }

  while(true) {
    const char C = upChar(*c);

    if (('0' <= *c) && (*c <= '9')) {
      ++digits;
    } else if (*c == '.') {
      decimal = true;
    } else {
      break;
    }
    ++c;
  }

  while(*c != '\0') {
    const char C = upChar(*c);

    if (C == 'L') {
      ++longs;
      ++c;
    } else if (C == 'U') {
      unsigned_ = true;
      ++c;
    } else if (C == 'E') {
      primitive exp = primitive::load(++c);
      // Check if there was an 'F' in exp
      float_ = (exp.type & ::isFloat);
      break;
    } else if (C == 'F') {
      float_ = true;
      ++c;
    } else {
      break;
    }
  }
  // If there was something else or no number
  if ((digits == 0) ||
      ((*c != '\0') && !charIsIn(*c, wordDelimiter1))) {
    p = (void*) NULL;
    p.type = none;
    c = c0;
    return p;
  }

  if (decimal || float_) {
    if (float_) {
      p = (float) occa::atof(std::string(c0, c - c0));
    } else {
      p = (double) occa::atod(std::string(c0, c - c0));
    }
  } else {
    uint64_t value = occa::atoi(std::string(c0, c - c0));
    if (longs == 0) {
      if (unsigned_) {
        p = (uint32_t) value;
      } else {
        p = (int32_t) value;
      }
    } else if (longs >= 1) {
      if (unsigned_) {
        p = (uint64_t) value;
      } else {
        p = (int64_t) value;
      }
    }
  }

  return p;
}

primitive primitive::load(const std::string &s) {
  const char *c = s.c_str();
  return load(c);
}

primitive primitive::loadBinary(const char *&c, const bool isNegative) {
  const char *c0 = c;
  uint64_t value = 0;
  while (*c == '0' || *c == '1') {
    value = (value << 1) | (*c - '0');
    ++c;
  }

  const int bits = c - c0 + isNegative;
  if (bits < 8) {
    return isNegative ? primitive((int8_t) -value) : primitive((uint8_t) value);
  } else if (bits < 16) {
    return isNegative ? primitive((int16_t) -value) : primitive((uint16_t) value);
  } else if (bits < 32) {
    return isNegative ? primitive((int32_t) -value) : primitive((uint32_t) value);
  }
  return isNegative ? primitive((int64_t) -value) : primitive((uint64_t) value);
}

primitive primitive::loadHex(const char *&c, const bool isNegative) {
  const char *c0 = c;
  uint64_t value = 0;
  while (true) {
    const char C = upChar(*c);
    if (('0' <= C) && (C <= '9')) {
      value = (value << 4) | (C - '0');
    } else if (('A' <= C) && (C <= 'F')) {
      value = (value << 4) | (10 + C - 'A');
    } else {
      break;
    }
    ++c;
  }

  const int bits = 4*(c - c0) + isNegative;
  if (bits < 8) {
    return isNegative ? primitive((int8_t) -value) : primitive((uint8_t) value);
  } else if (bits < 16) {
    return isNegative ? primitive((int16_t) -value) : primitive((uint16_t) value);
  } else if (bits < 32) {
    return isNegative ? primitive((int32_t) -value) : primitive((uint32_t) value);
  }
  return isNegative ? primitive((int64_t) -value) : primitive((uint64_t) value);
}

//---[ Unary Operators ]----------------
primitive primitive::operator ! () {
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

primitive primitive::operator + () {
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

primitive primitive::operator - () {
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

primitive primitive::operator ~ () {
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

primitive& primitive::operator ++ () {
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

primitive& primitive::operator -- () {
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

primitive primitive::operator ++ (int) {
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

primitive primitive::operator -- (int) {
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
//======================================


//---[ Boolean Operators ]--------------
primitive primitive::operator < (const primitive &p) {
  const type_t retType = (type > p.type) ? type : p.type;
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

primitive primitive::operator == (const primitive &p) {
  const type_t retType = (type > p.type) ? type : p.type;
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

primitive primitive::operator != (const primitive &p) {
  const type_t retType = (type > p.type) ? type : p.type;
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

primitive primitive::operator >= (const primitive &p) {
  const type_t retType = (type > p.type) ? type : p.type;
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

primitive primitive::operator > (const primitive &p) {
  const type_t retType = (type > p.type) ? type : p.type;
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

primitive primitive::operator && (const primitive &p) {
  const type_t retType = (type > p.type) ? type : p.type;
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

primitive primitive::operator || (const primitive &p) {
  const type_t retType = (type > p.type) ? type : p.type;
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
//======================================


//---[ Binary Operators ]---------------
primitive primitive::operator * (const primitive &p) {
  const type_t retType = (type > p.type) ? type : p.type;
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

primitive primitive::operator + (const primitive &p) {
  const type_t retType = (type > p.type) ? type : p.type;
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

primitive primitive::operator - (const primitive &p) {
  const type_t retType = (type > p.type) ? type : p.type;
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

primitive primitive::operator / (const primitive &p) {
  const type_t retType = (type > p.type) ? type : p.type;
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

primitive primitive::operator % (const primitive &p) {
  const type_t retType = (type > p.type) ? type : p.type;
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

primitive primitive::operator & (const primitive &p) {
  const type_t retType = (type > p.type) ? type : p.type;
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

primitive primitive::operator | (const primitive &p) {
  const type_t retType = (type > p.type) ? type : p.type;
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

primitive primitive::operator ^ (const primitive &p) {
  const type_t retType = (type > p.type) ? type : p.type;
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

primitive primitive::operator >> (const primitive &p) {
  const type_t retType = (type > p.type) ? type : p.type;
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

primitive primitive::operator << (const primitive &p) {
  const type_t retType = (type > p.type) ? type : p.type;
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
//======================================


//---[ Assignment Operators ]-----------
primitive primitive::operator *= (const primitive &p) {
 *this = (*this * p);
 return *this;
}

primitive primitive::operator += (const primitive &p) {
 *this = (*this + p);
 return *this;
}

primitive primitive::operator -= (const primitive &p) {
 *this = (*this - p);
 return *this;
}

primitive primitive::operator /= (const primitive &p) {
 *this = (*this / p);
 return *this;
}

primitive primitive::operator %= (const primitive &p) {
 *this = (*this % p);
 return *this;
}

primitive primitive::operator &= (const primitive &p) {
 *this = (*this & p);
 return *this;
}

primitive primitive::operator |= (const primitive &p) {
 *this = (*this | p);
 return *this;
}

primitive primitive::operator ^= (const primitive &p) {
 *this = (*this ^ p);
 return *this;
}

primitive primitive::operator >>= (const primitive &p) {
 *this = (*this >> p);
 return *this;
}

primitive primitive::operator <<= (const primitive &p) {
 *this = (*this << p);
 return *this;
}
//======================================

primitive::operator std::string () const {
  std::string str;
  switch(type) {
  case uint8_  : str = occa::toString((uint64_t) value.uint8_);  break;
  case uint16_ : str = occa::toString((uint64_t) value.uint16_); break;
  case uint32_ : str = occa::toString((uint64_t) value.uint32_); break;
  case uint64_ : str = occa::toString((uint64_t) value.uint64_); break;
  case int8_   : str = occa::toString((int64_t)  value.int8_);   break;
  case int16_  : str = occa::toString((int64_t)  value.int16_);  break;
  case int32_  : str = occa::toString((int64_t)  value.int32_);  break;
  case int64_  : str = occa::toString((int64_t)  value.int64_);  break;
  case float_  : str = occa::toString(value.float_);  break;
  case double_ : str = occa::toString(value.double_); break;
  default: OCCA_FORCE_ERROR("Type not set");
  }

  if ((str.find("inf") != std::string::npos) ||
      (str.find("INF") != std::string::npos)) {
    return str;
  }

  if (type & (uint64_ | int64_)) {
    str += 'L';
  }
  return str;
}

std::ostream& operator << (std::ostream &out, const primitive &p) {
  out << (std::string) p;
  return out;
}