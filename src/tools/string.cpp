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

#include "occa/tools/lex.hpp"
#include "occa/tools/string.hpp"
#include "occa/tools/sys.hpp"
#include "occa/parser/tools.hpp"

namespace occa {
  std::string strip(const std::string &str) {
    const char *start = str.c_str();
    const char *end   = start + str.size();
    lex::strip(start, end);
    return std::string(start, end - start);
  }

  std::string escape(const std::string &str, const char c, const char escapeChar) {
    const int chars = (int) str.size();
    const char *cstr = str.c_str();
    std::string ret;
    for (int i = 0; i < chars; ++i) {
      if (cstr[i] == c) {
        ret += escapeChar;
      }
      ret += cstr[i];
    }
    return ret;
  }

  std::string unescape(const std::string &str, const char c, const char escapeChar) {
    std::string ret;
    const int chars = (int) str.size();
    const char *cstr = str.c_str();
    for (int i = 0; i < chars; ++i) {
      if (cstr[i] == escapeChar && cstr[i + 1] == c) {
        continue;
      }
      ret += cstr[i];
    }
    return ret;
  }

  strVector_t split(const std::string &s, const char delimeter, const char escapeChar) {
    strVector_t sv;
    const char *c = s.c_str();

    while (*c != '\0') {
      const char *cStart = c;
      skipTo(c, delimeter, escapeChar);
      sv.push_back(std::string(cStart, c - cStart));
      if (*c != '\0') {
        ++c;
      }
    }

    return sv;
  }

  std::string uppercase(const char *c, const int chars) {
    std::string ret(c, chars);
    for (int i = 0; i < chars; ++i) {
      ret[i] = uppercase(ret[i]);
    }
    return ret;
  }

  std::string lowercase(const char *c, const int chars) {
    std::string ret(c, chars);
    for (int i = 0; i < chars; ++i) {
      ret[i] = lowercase(ret[i]);
    }
    return ret;
  }

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

        OCCA_ERROR("Number [" << std::string(c0, c - c0)
                   << "...] must contain digits in the [0,"
                   << maxDigitChar << "] range",
                   digitValue < maxDigitValue);

        ret <<= bits;
        ret += digitValue;
      } else {
        const char C = upChar(*c);

        if (('A' <= C) && (C <= 'F')) {
          const char digitValue = 10 + (C - 'A');
          ++c;

          OCCA_ERROR("Number [" << std::string(c0, c - c0)
                     << "...] must contain digits in the [0,"
                     << maxDigitChar << "] range",
                     digitValue < maxDigitValue);

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

  //---[ Vector Methods ]---------------
  std::string join(const strVector_t &vec, const std::string &seq) {
    const int entries = (int) vec.size();
    if (entries == 0) {
      return "";
    }
    std::string ret = vec[0];
    for (int i = 1; i < entries; ++i) {
      ret += seq;
      ret += vec[1];
    }
    return ret;
  }
  //====================================

  //---[ Color Strings ]----------------
  namespace color {
    const char fgMap[9][7] = {
      "\033[39m",
      "\033[30m", "\033[31m", "\033[32m", "\033[33m",
      "\033[34m", "\033[35m", "\033[36m", "\033[37m"
    };

    const char bgMap[9][7] = {
      "\033[49m",
      "\033[40m", "\033[41m", "\033[42m", "\033[43m",
      "\033[44m", "\033[45m", "\033[46m", "\033[47m"
    };

    std::string string(const std::string &s, color_t fg) {
      std::string ret = fgMap[fg];
      ret += s;
      ret += fgMap[normal];
      return ret;
    }

    std::string string(const std::string &s, color_t fg, color_t bg) {
      std::string ret = fgMap[fg];
      ret += bgMap[bg];
      ret += s;
      ret += fgMap[normal];
      ret += bgMap[normal];
      return ret;
    }
  }

  std::string black(const std::string &s) {
    return color::string(s, color::black);
  }

  std::string red(const std::string &s) {
    return color::string(s, color::red);
  }

  std::string green(const std::string &s) {
    return color::string(s, color::green);
  }

  std::string yellow(const std::string &s) {
    return color::string(s, color::yellow);
  }

  std::string blue(const std::string &s) {
    return color::string(s, color::blue);
  }

  std::string magenta(const std::string &s) {
    return color::string(s, color::magenta);
  }

  std::string cyan(const std::string &s) {
    return color::string(s, color::cyan);
  }

  std::string white(const std::string &s) {
    return color::string(s, color::white);
  }
  //====================================
}
