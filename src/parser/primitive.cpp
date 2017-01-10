/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2016 David Medina and Tim Warburton
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

#include <cstring>

#include "occa/parser/primitive.hpp"

namespace occa {
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

      const char C = uppercase(*c);

      if (C == 'B') {
        return primitive::loadBinary(++c, negative);
      } else if (C == 'X') {
        return primitive::loadHex(++c, negative);
      } else {
        --c;
      }
    }

    while(true) {
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
      const char C = uppercase(*c);

      if (C == 'L') {
        ++longs;
        ++c;
      } else if (C == 'U') {
        unsigned_ = true;
        ++c;
      } else if (C == 'E') {
        primitive exp = primitive::load(++c);
        // Check if there was an 'F' in exp
        float_ = (exp.type & isFloat);
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
        ((*c != '\0') && !charIsDelimiter(*c))) {
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
      const char C = uppercase(*c);
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
}
