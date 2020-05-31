#include <occa/defines.hpp>

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
#  include <stdio.h>
#else // OCCA_WINDOWS_OS
#  include <windows.h>
#endif

#include <occa/tools/env.hpp>
#include <occa/tools/lex.hpp>
#include <occa/tools/string.hpp>
#include <occa/tools/sys.hpp>

namespace occa {
  std::string strip(const std::string &str) {
    std::string output;
    strip(str, true, true, output);
    return output;
  }

  std::string stripLeft(const std::string &str) {
    std::string output;
    strip(str, true, false, output);
    return output;
  }

  std::string stripRight(const std::string &str) {
    std::string output;
    strip(str, false, true, output);
    return output;
  }

  void strip(const std::string &str,
             const bool stripLeft,
             const bool stripRight,
             std::string &output) {
    const char *start = str.c_str();
    const char *end = start + str.size() - 1;

    if (start >= end) {
      output = str;
      return;
    }

    if (stripLeft) {
      while ((*start != '\0') &&
             lex::isWhitespace(*start)) {
        ++start;
      }
    }

    if (stripRight) {
      while ((start < end) &&
             lex::isWhitespace(*end)) {
        --end;
      }
      ++end;
    }

    output = std::string(start, end - start);
  }

  std::string escape(const std::string &str,
                     const char c,
                     const char escapeChar) {
    const int chars = (int) str.size();
    const char *cstr = str.c_str();
    std::string ret;
    for (int i = 0; i < chars; ++i) {
      if (cstr[i] != c) {
        ret += cstr[i];
      } else {
        if (i && escapeChar) {
          ret += escapeChar;
        }
        ret += c;
      }
    }
    return ret;
  }

  std::string unescape(const std::string &str,
                       const char c,
                       const char escapeChar) {
    std::string ret;
    const int chars = (int) str.size();
    const char *cstr = str.c_str();
    for (int i = 0; i < chars; ++i) {
      if (escapeChar &&
          (cstr[i] == escapeChar) &&
          (cstr[i + 1] == c)) {
        continue;
      }
      ret += cstr[i];
    }
    return ret;
  }

  strVector split(const std::string &s,
                  const char delimeter,
                  const char escapeChar) {
    strVector sv;
    const char *c = s.c_str();

    while (*c != '\0') {
      const char *cStart = c;
      lex::skipTo(c, delimeter, escapeChar);
      sv.push_back(std::string(cStart, c - cStart));
      if (*c != '\0') {
        ++c;
      }
    }

    return sv;
  }

  std::string uppercase(const char *c,
                        const int chars) {
    std::string ret(c, chars);
    for (int i = 0; i < chars; ++i) {
      ret[i] = uppercase(ret[i]);
    }
    return ret;
  }

  std::string lowercase(const char *c,
                        const int chars) {
    std::string ret(chars, '\0');
    for (int i = 0; i < chars; ++i) {
      ret[i] = lowercase(c[i]);
    }
    return ret;
  }

  template <>
  std::string toString<std::string>(const std::string &t) {
    return t;
  }

  template <>
  std::string toString<bool>(const bool &t) {
    return t ? "true" : "false";
  }

  template <>
  std::string toString<float>(const float &t) {
    std::stringstream ss;
    ss << std::scientific << std::setprecision(8) << t << 'f';
    return ss.str();
  }

  template <>
  std::string toString<double>(const double &t) {
    std::stringstream ss;
    ss << std::scientific << std::setprecision(16) << t;
    return ss.str();
  }

  template <>
  bool fromString(const std::string &s) {
    if (s == "0") {
      return false;
    }
    const std::string sUp = uppercase(s);
    return !((sUp == "N") ||
             (sUp == "NO") ||
             (sUp == "FALSE"));
  }

  template <>
  std::string fromString(const std::string &s) {
    return s;
  }

  udim_t atoi(const char *c) {
    udim_t ret = 0;

    const char *c0 = c;

    bool negative  = false;
    bool unsigned_ = false;
    int longs      = 0;

    lex::skipWhitespace(c);

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
      const char C = uppercase(*c);

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

#if !OCCA_UNSAFE
    const char *c0 = c;
    int maxDigitValue = 10;
    char maxDigitChar = '9';
#endif

    bool negative = false;
    int bits = 3;

    lex::skipWhitespace(c);

    if ((*c == '+') || (*c == '-')) {
      negative = (*c == '-');
      ++c;
    }

    if (*c == '0') {
      ++c;

      const char C = uppercase(*c);

      if (C == 'X') {
        bits = 4;
        ++c;
#if !OCCA_UNSAFE
        maxDigitValue = 16;
        maxDigitChar  = 'F';
#endif
      } else if (C == 'B') {
        bits = 1;
        ++c;
#if !OCCA_UNSAFE
        maxDigitValue = 2;
        maxDigitChar  = '1';
#endif
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
        const char C = uppercase(*c);

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
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
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

      if (bigBytes < (((uint64_t) 1) << 10)) {
        ss << bigBytes << " bytes";
      } else if (bigBytes < (((uint64_t) 1) << 20)) {
        ss << (bigBytes >> 10);
        stringifyBytesFraction(ss, bigBytes >> 0);
        ss << " KB";
      } else if (bigBytes < (((uint64_t) 1) << 30)) {
        ss << (bigBytes >> 20);
        stringifyBytesFraction(ss, bigBytes >> 10);
        ss << " MB";
      } else if (bigBytes < (((uint64_t) 1) << 40)) {
        ss << (bigBytes >> 30);
        stringifyBytesFraction(ss, bigBytes >> 20);
        ss << " GB";
      } else if (bigBytes < (((uint64_t) 1) << 50)) {
        ss << (bigBytes >> 40);
        stringifyBytesFraction(ss, bigBytes >> 30);
        ss << " TB";
      } else {
        ss << bigBytes << " bytes";
      }
      return ss.str();
    }

    return "";
  }

  void stringifyBytesFraction(std::stringstream &ss,
                              uint64_t fraction) {
    const int part = (int) (100.0 * (fraction % 1024) / 1024.0);
    if (part) {
      ss << '.' << part;
    }
  }

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

    std::string string(const std::string &s,
                       color_t fg) {
      if (!env::OCCA_COLOR_ENABLED) {
        return s;
      }
      std::string ret = fgMap[fg];
      ret += s;
      ret += fgMap[normal];
      return ret;
    }

    std::string string(const std::string &s,
                       color_t fg,
                       color_t bg) {
      if (!env::OCCA_COLOR_ENABLED) {
        return s;
      }
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
