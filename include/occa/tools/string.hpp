#ifndef OCCA_TOOLS_STRING_HEADER
#define OCCA_TOOLS_STRING_HEADER

#include <iostream>
#include <iomanip>
#include <sstream>

#include <cstring>
#include <cstdlib>

#include <occa/defines.hpp>
#include <occa/types.hpp>

namespace occa {
  std::string strip(const std::string &str);

  std::string stripLeft(const std::string &str);

  std::string stripRight(const std::string &str);

  void strip(const std::string &str,
             const bool stripLeft,
             const bool stripRight,
             std::string &output);

  std::string escape(const std::string &str,
                     const char c,
                     const char excapeChar = '\\');

  std::string unescape(const std::string &str,
                       const char c,
                       const char excapeChar = '\\');

  strVector split(const std::string &s,
                  const char delimeter,
                  const char escapeChar = 0);

  inline char uppercase(const char c) {
    if (('a' <= c) && (c <= 'z')) {
      return ((c + 'A') - 'a');
    }
    return c;
  }

  inline char lowercase(const char c) {
    if (('A' <= c) && (c <= 'Z')) {
      return ((c + 'a') - 'A');
    }
    return c;
  }

  std::string uppercase(const char *c,
                        const int chars);

  inline std::string uppercase(const std::string &s) {
    return uppercase(s.c_str(), s.size());
  }

  std::string lowercase(const char *c,
                        const int chars);

  inline std::string lowercase(const std::string &s) {
    return lowercase(s.c_str(), s.size());
  }

  template <class TM>
  std::string toString(const TM &t) {
    std::stringstream ss;
    ss << t;
    return ss.str();
  }

  template <class TM>
  std::string toString(const std::vector<TM> &v) {
    const int size = (int) v.size();
    std::stringstream ss;
    ss << '[';
    for (int i = 0; i < size; ++i) {
      const std::string istr = occa::toString(v[i]);
      ss << escape(istr, ',');
      if (i < (size - 1)) {
        ss << ',';
      }
    }
    ss << ']';
    return ss.str();
  }

  template <>
  std::string toString<std::string>(const std::string &t);

  template <>
  std::string toString<bool>(const bool &t);

  template <>
  std::string toString<float>(const float &t);

  template <>
  std::string toString<double>(const double &t);

  template <class TM>
  TM fromString(const std::string &s) {
    std::stringstream ss;
    TM t;
    ss << s;
    ss >> t;
    return t;
  }

  template <>
  bool fromString(const std::string &s);

  template <>
  std::string fromString(const std::string &s);

  template <class TM>
  std::vector<TM> listFromString(const std::string &s) {
    std::string str = strip(s);
    const int chars = (int) str.size();
    if (chars && str[chars - 1] == ']') {
      str = str.substr(0, chars - 1);
    }
    if (chars && str[0] == '[') {
      str = str.substr(1);
    }

    strVector parts = split(str, ',', '\\');
    const int partCount = (int) parts.size();

    std::vector<TM> ret;
    ret.reserve(partCount);
    for (int i = 0; i < partCount; ++i) {
      ret.push_back(occa::fromString<TM>(unescape(parts[i], '\\')));
    }
    return ret;
  }

  inline bool startsWith(const std::string &s,
                         const std::string &match) {
    const int matchChars = (int) match.size();
    return ((matchChars <= (int) s.size()) &&
            (!strncmp(s.c_str(), match.c_str(), matchChars)));
  }

  inline bool endsWith(const std::string &s,
                       const std::string &match) {
    const int sChars = (int) s.size();
    const int matchChars = (int) match.size();
    return ((matchChars <= sChars) &&
            (!strncmp(s.c_str() + (sChars - matchChars),
                      match.c_str(),
                      matchChars)));
  }

  udim_t atoi(const char *c);
  udim_t atoiBase2(const char *c);
  udim_t atoi(const std::string &str);

  double atof(const char *c);
  double atof(const std::string &str);

  double atod(const char *c);
  double atod(const std::string &str);

  inline char toHexChar(const char c) {
    if (c < 16) {
      // 'a' = 'W' + 10
      return c + ((c < 10) ? '0' : 'W');
    }
    return c;
  }

  inline char fromHexChar(const char c) {
    if (('0' <= c) && (c <= '9')) {
      return c - '0';
    }
    if (('a' <= c) && (c <= 'z')) {
      return 10 + (c - 'a');
    }
    if (('A' <= c) && (c <= 'Z')) {
      return 10 + (c - 'A');
    }
    return c;
  }

  template <class TM>
  std::string toHex(const TM &value) {
    std::string str;
    const char *c = (const char*) &value;
    const int bytes = (int) sizeof(value);

    for (int i = 0; i < bytes; ++i) {
      const char ci = c[i];
      str += toHexChar((ci >> 4) & 0xF);
      str += toHexChar(ci        & 0xF);
    }

    return str;
  }

  template <class TM>
  void fromHex(const std::string &str,
               TM *out,
               const int bytes) {
    const int chars = (int) str.size();
    const int hexChars = ((chars > (2 * bytes))
                          ? bytes
                          : (chars / 2));

    ::memset(out, 0, bytes);

    const char *c_in = str.c_str();
    char *c_out = (char*) out;

    for (int i = 0; i < hexChars; ++i) {
      const char c1 = fromHexChar(c_in[2*i + 0]);
      const char c2 = fromHexChar(c_in[2*i + 1]);
      c_out[i] = (c1 << 4) | c2;
    }
  }

  template <class TM>
  TM fromHex(const std::string &str) {
    TM value;
    fromHex(str, &value, sizeof(value));
    return value;
  }

  template <class TM>
  std::string stringifySetBits(const TM value) {
    if (value == 0) {
      return "No bits set";
    }
    std::stringstream ss;
    const int bits = (int) (8 * sizeof(TM));
    bool hasBits = false;
    for (int i = 0; i < bits; ++i) {
      if (value & (((TM) 1) << i)) {
        if (hasBits) {
          ss << ", ";
        }
        ss << i;
        hasBits = true;
      }
    }
    return ss.str();
  }

  std::string stringifyBytes(udim_t bytes);

  void stringifyBytesFraction(std::stringstream &ss,
                              uint64_t fraction);

  //---[ Color Strings ]----------------
  namespace color {
    enum color_t {
      normal  = 0,
      black   = 1,
      red     = 2,
      green   = 3,
      yellow  = 4,
      blue    = 5,
      magenta = 6,
      cyan    = 7,
      white   = 8
    };

    extern const char fgMap[9][7];
    extern const char bgMap[9][7];

    std::string string(const std::string &s,
                       color_t fg);
    std::string string(const std::string &s,
                       color_t fg,
                       color_t bg);
  }

  std::string black(const std::string &s);
  std::string red(const std::string &s);
  std::string green(const std::string &s);
  std::string yellow(const std::string &s);
  std::string blue(const std::string &s);
  std::string magenta(const std::string &s);
  std::string cyan(const std::string &s);
  std::string white(const std::string &s);
  //====================================
}

#endif
