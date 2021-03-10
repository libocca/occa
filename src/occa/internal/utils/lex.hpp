#ifndef OCCA_INTERNAL_UTILS_LEX_HEADER
#define OCCA_INTERNAL_UTILS_LEX_HEADER

#include <iostream>
#include <iomanip>
#include <sstream>

#include <occa/defines.hpp>
#include <occa/types.hpp>

namespace occa {
  namespace lex {
    extern const char whitespaceCharset[];
    extern const char numberCharset[];

    inline bool isDigit(const char c) {
      return (('0' <= c) && (c <= '9'));
    }

    inline bool isAlpha(const char c) {
      return ((('a' <= c) && (c <= 'z')) ||
              (('A' <= c) && (c <= 'Z')));
    }

    bool inCharset(const char c, const char *charset);

    //---[ Skip ]-----------------------
    void skipTo(const char *&c, const char delimiter);
    void skipTo(const char *&c, const char delimiter, const char escapeChar);
    void skipTo(const char *&c, const char *delimiters);
    void skipTo(const char *&c, const char *delimiters, const char escapeChar);

    void skipFrom(const char *&c, const char *delimiters);
    //==================================

    //---[ Whitespace ]-----------------
    bool isWhitespace(const char c);

    void skipWhitespace(const char *&c);

    void skipToWhitespace(const char *&c);
    //==================================
  }
}
#endif
