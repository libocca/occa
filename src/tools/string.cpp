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

#include "occa/tools/string.hpp"
#include "occa/parser/tools.hpp"

namespace occa {
  udim_t atoi(const char *c) {
    udim_t ret = 0;

    const char *c0 = c;

    bool negative  = false;
    bool unsigned_ = false;
    int longs      = 0;

    skipWhitespace(c);

    if ((*c == '+') || (*c == '-')) {
      negative = (*c == '-');
      ++c;
    }

    if (c[0] == '0')
      return atoiBase2(c0);

    while(('0' <= *c) && (*c <= '9')) {
      ret *= 10;
      ret += *(c++) - '0';
    }

    while(*c != '\0') {
      const char C = upChar(*c);

      if (C == 'L') {
        ++longs;
      } else if (C == 'U') {
        unsigned_ = true;
      } else {
        break;
      }
      ++c;
    }

    if (negative) {
      ret = ((~ret) + 1);
    }
    if (longs == 0) {
      if (!unsigned_) {
        ret = ((udim_t) ((int) ret));
      } else {
        ret = ((udim_t) ((unsigned int) ret));
      }
    } else if (longs == 1) {
      if (!unsigned_) {
        ret = ((udim_t) ((long) ret));
      } else {
        ret = ((udim_t) ((unsigned long) ret));
      }
    }
    // else it's already in udim_t form

    return ret;
  }

  udim_t atoiBase2(const char*c) {
    udim_t ret = 0;

    const char *c0 = c;

    bool negative     = false;
    int bits          = 3;
    int maxDigitValue = 10;
    char maxDigitChar = '9';

    skipWhitespace(c);

    if ((*c == '+') || (*c == '-')) {
      negative = (*c == '-');
      ++c;
    }

    if (*c == '0') {
      ++c;

      const char C = upChar(*c);

      if (C == 'X') {
        bits = 4;
        ++c;

        maxDigitValue = 16;
        maxDigitChar  = 'F';
      } else if (C == 'B') {
        bits = 1;
        ++c;

        maxDigitValue = 2;
        maxDigitChar  = '1';
      }
    }

    while(true) {
      if (('0' <= *c) && (*c <= '9')) {
        const char digitValue = *(c++) - '0';

        OCCA_CHECK(digitValue < maxDigitValue,
                   "Number [" << std::string(c0, c - c0)
                   << "...] must contain digits in the [0,"
                   << maxDigitChar << "] range");

        ret <<= bits;
        ret += digitValue;
      } else {
        const char C = upChar(*c);

        if (('A' <= C) && (C <= 'F')) {
          const char digitValue = 10 + (C - 'A');
          ++c;

          OCCA_CHECK(digitValue < maxDigitValue,
                     "Number [" << std::string(c0, c - c0)
                     << "...] must contain digits in the [0,"
                     << maxDigitChar << "] range");

          ret <<= bits;
          ret += digitValue;
        } else {
          break;
        }
      }
    }

    if (negative) {
      ret = ((~ret) + 1);
    }
    return ret;
  }

  udim_t atoi(const std::string &str) {
    return occa::atoi((const char*) str.c_str());
  }

  double atof(const char *c) {
    return ::atof(c);
  }

  double atof(const std::string &str) {
    return ::atof(str.c_str());
  }

  double atod(const char *c) {
    double ret;
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
    sscanf(c, "%lf", &ret);
#else
    sscanf_s(c, "%lf", &ret);
#endif
    return ret;
  }

  double atod(const std::string &str) {
    return occa::atod(str.c_str());
  }

  std::string stringifyBytes(udim_t bytes) {
    if (0 < bytes) {
      std::stringstream ss;
      uint64_t bigBytes = bytes;
      uint64_t big1 = 1;

      if (bigBytes < (big1 << 10)) {
        ss << bigBytes << " bytes";
      } else if (bigBytes < (big1 << 20)) {
        ss << (bigBytes >> 10) << " KB";
      } else if (bigBytes < (big1 << 30)) {
        ss << (bigBytes >> 20) << " MB";
      } else if (bigBytes < (big1 << 40)) {
        ss << (bigBytes >> 30) << " GB";
      } else if (bigBytes < (big1 << 50)) {
        ss << (bigBytes >> 40) << " TB";
      } else {
        ss << bigBytes << " bytes";
      }
      return ss.str();
    }

    return "";
  }

  strVector_t split(const std::string &s, const char delimeter) {
    strVector_t sv;
    const char *c = s.c_str();

    while (*c != '\0') {
      const char *cStart = c;
      skipTo(c, delimeter);
      sv.push_back(std::string(cStart, c - cStart));
      if (*c != '\0') {
        ++c;
      }
    }

    return sv;
  }
}
