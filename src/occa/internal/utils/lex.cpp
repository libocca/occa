#include <occa/internal/utils/lex.hpp>

namespace occa {
  namespace lex {
    const char whitespaceCharset[] = " \t\r\n\v\f";
    const char numberCharset[]     = "0123456789";

    bool inCharset(const char c, const char *charset) {
      while (*charset != '\0') {
        if (c == *(charset++)) {
          return true;
        }
      }
      return false;
    }

    //---[ Skip ]-----------------------
    void skipTo(const char *&c, const char delimiter) {
      while (*c != '\0') {
        if (*c == delimiter) {
          return;
        }
        ++c;
      }
    }

    void skipTo(const char *&c, const char delimiter, const char escapeChar) {
      while (*c != '\0') {
        if (escapeChar &&
            (*c == escapeChar)) {
          c += 1 + (c[1] != '\0');
          continue;
        }
        if (*c == delimiter) {
          return;
        }
        ++c;
      }
    }

    void skipTo(const char *&c, const char *delimiters) {
      while (*c != '\0') {
        if (inCharset(*c, delimiters)) {
          return;
        }
        ++c;
      }
    }

    void skipTo(const char *&c, const char *delimiters, const char escapeChar) {
      while (*c != '\0') {
        if (escapeChar &&
            (*c == escapeChar)) {
          c += 1 + (c[1] != '\0');
          continue;
        }
        if (inCharset(*c, delimiters)) {
          return;
        }
        ++c;
      }
    }

    void skipFrom(const char *&c, const char *delimiters) {
      while (*c != '\0') {
        if (inCharset(*c, delimiters)) {
          ++c;
          continue;
        }
        return;
      }
    }
    //==================================

    //---[ Whitespace ]-----------------
    bool isWhitespace(const char c) {
      return inCharset(c, whitespaceCharset);
    }

    void skipWhitespace(const char *&c) {
      skipFrom(c, whitespaceCharset);
    }

    void skipToWhitespace(const char *&c) {
      skipTo(c, whitespaceCharset);
    }
  }
}
