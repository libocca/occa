/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
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

namespace occa {
  namespace lex {
    const char whitespaceChars[] = " \t\r\n\v\f";
    const char alphaChars[]    = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const char numChars[]      = "0123456789";
    const char alphanumChars[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    const char identifierStartChar[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_";
    const char identifierChars[]     = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789";

    const char quote1Delimiters[] = "\n'";
    const char quote2Delimiters[] = "\n\"";

    bool charIsIn(const char c, const char *delimiters) {
      while (*delimiters != '\0') {
        if (c == *(delimiters++)) {
          return true;
        }
      }
      return false;
    }

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
        if (*c == escapeChar) {
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
        if (charIsIn(*c, delimiters)) {
          return;
        }
        ++c;
      }
    }

    void skipTo(const char *&c, const char *delimiters, const char escapeChar) {
      while (*c != '\0') {
        if (*c == escapeChar) {
          c += 1 + (c[1] != '\0');
          continue;
        }
        if (charIsIn(*c, delimiters)) {
          return;
        }
        ++c;
      }
    }

    void skipTo(const char *&c, const std::string &delimiters) {
      skipTo(c, delimiters.c_str());
    }

    void skipTo(const char *&c, const std::string &delimiters, const char escapeChar) {
      skipTo(c, delimiters.c_str(), escapeChar);
    }

    void quotedSkipTo(const char *&c, const char delimiter) {
      while (*c != '\0') {
        if (*c == '\\') {
          c += 1 + (c[1] != '\0');
          continue;
        }
        if (*c == delimiter) {
          return;
        } else if (*c == '\'') {
          ++c;
          skipTo(c, quote1Delimiters, '\\');
          c += (*c == '\'');
        } else if (*c == '"') {
          ++c;
          skipTo(c, quote2Delimiters, '\\');
          c += (*c == '"');
        } else {
          ++c;
        }
      }
    }

    void quotedSkipTo(const char *&c, const char *delimiters) {
      while (*c != '\0') {
        if (*c == '\\') {
          c += 1 + (c[1] != '\0');
          continue;
        }
        if (charIsIn(*c, delimiters)) {
          return;
        } else if (*c == '\'') {
          ++c;
          skipTo(c, quote1Delimiters, '\\');
          c += (*c == '\'');
        } else if (*c == '"') {
          ++c;
          skipTo(c, quote2Delimiters, '\\');
          c += (*c == '"');
        }
        ++c;
      }
    }

    void quotedSkipTo(const char *&c, const std::string &delimiters) {
      quotedSkipTo(c, delimiters.c_str());
    }

    void skipToMatch(const char *&c, const std::string &match) {
      const size_t chars = match.size();
      const char *d      = match.c_str();

      while (*c != '\0') {
        for (size_t i = 0; i < chars; ++i) {
          if (c[i] != d[i]) {
            continue;
          }
          return;
        }
        ++c;
      }
    }

    void skipToMatch(const char *&c, const std::string &match, const char escapeChar) {
      const size_t chars = match.size();
      const char *d      = match.c_str();

      while (*c != '\0') {
        if (*c == escapeChar) {
          c += 1 + (c[1] != '\0');
          continue;
        }
        for (size_t i = 0; i < chars; ++i) {
          if (c[i] != d[i]) {
            continue;
          }
          return;
        }
        ++c;
      }
    }

    void skipFrom(const char *&c, const char *delimiters) {
      while (*c != '\0') {
        if (charIsIn(*c, delimiters)) {
          ++c;
          continue;
        }
        return;
      }
    }

    void skipFrom(const char *&c, const char *delimiters, const char escapeChar) {
      while (*c != '\0') {
        if (*c == escapeChar) {
          c += 1 + (c[1] != '\0');
          continue;
        }
        if (charIsIn(*c, delimiters)) {
          ++c;
          continue;
        }
        return;
      }
    }

    void skipFrom(const char *&c, const std::string &delimiters) {
      skipFrom(c, delimiters.c_str());
    }

    void skipFrom(const char *&c, const std::string &delimiters, const char escapeChar) {
      skipFrom(c, delimiters.c_str(), escapeChar);
    }

    bool isWhitespace(const char c) {
      return charIsIn(c, whitespaceChars);
    }

    void skipWhitespace(const char *&c) {
      skipFrom(c, whitespaceChars);
    }

    void skipWhitespace(const char *&c, const char escapeChar) {
      skipFrom(c, whitespaceChars, escapeChar);
    }

    void skipToWhitespace(const char *&c) {
      skipTo(c, whitespaceChars);
    }

    void skipBetweenWhitespaces(const char *&c) {
      skipWhitespace(c);
      skipToWhitespace(c);
      skipWhitespace(c);
    }

    void skipBetweenWhitespaces(const char *&c, const char escapeChar) {
      skipWhitespace(c, escapeChar);
      skipToWhitespace(c);
      skipWhitespace(c, escapeChar);
    }

    void strip(const char *&start, const char *&end) {
      if (end <= start) {
        return;
      }
      while ((*start != '\0') &&
             isWhitespace(*start)) {
        ++start;
      }
      while ((start < end) &&
             isWhitespace(*end)) {
        --end;
      }
    }

    void strip(const char *&start, const char *&end, const char escapeChar) {
      if (end <= start) {
        return;
      }
      while (*start != '\0') {
        if (isWhitespace(*start)) {
          ++start;
        } else if ((*start == escapeChar) &&
                   isWhitespace(start[1])) {
          start += 2;
        } else {
          break;
        }
      }
      while (start < end) {
        if (isWhitespace(*end)) {
          --end;
        } else if ((*end == escapeChar) &&
                   isWhitespace(end[1])) {
          end -= 2;
        } else {
          break;
        }
      }
    }
  }
}
