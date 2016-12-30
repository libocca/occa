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

namespace occa {
  namespace lex {
     const char whitespaceChars[] = " \t\r\n\v\f\0";

    void skipTo(const char *&c, const char delimiter) {
      while (*c != '\0') {
        if(*c == delimiter) {
          return;
        }
        ++c;
      }
    }

    void skipTo(const char *&c, const char delimiter, const char escapeChar) {
      while (*c != '\0') {
        if (*c == escapeChar) {
          c += 2;
          continue;
        }
        if(*c == delimiter) {
          return;
        }
        ++c;
      }
    }

    void skipTo(const char *&c, const std::string &match) {
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

    void skipTo(const char *&c, const std::string &match, const char escapeChar) {
      const size_t chars = match.size();
      const char *d      = match.c_str();

      while (*c != '\0') {
        if (*c == escapeChar) {
          c += 2;
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

    void skipToDelimiter(const char *&c, const std::string &delimiters) {
      const size_t chars = delimiters.size();
      const char *d      = delimiters.c_str();

      while (*c != '\0') {
        for (size_t i = 0; i < chars; ++i) {
          if (*c == d[i]) {
            return;
          }
        }
        ++c;
      }
    }

    void skipToDelimiter(const char *&c, const std::string &delimiters, const char escapeChar) {
      const size_t chars = delimiters.size();
      const char *d      = delimiters.c_str();

      while (*c != '\0') {
        if (*c == escapeChar) {
          c += 2;
          continue;
        }
        for (size_t i = 0; i < chars; ++i) {
          if (*c == d[i]) {
            return;
          }
        }
        ++c;
      }
    }

    bool charIsIn(const char c, const char *delimiters) {
      while (*delimiters != '\0') {
        if (c == *(delimiters++)) {
          return true;
        }
      }
      return false;
    }

    bool isWhitespace(const char c) {
      return charIsIn(c, whitespaceChars);
    }

    void skipWhitespace(const char *&c) {
      while (charIsIn(*c, whitespaceChars) &&
             (*c != '\0')) {
        ++c;
      }
    }

    void skipToWhitespace(const char *&c) {
      while (!charIsIn(*c, whitespaceChars) &&
             (*c != '\0')) {
        ++c;
      }
    }

    void skipBetweenWhitespaces(const char *&c) {
      skipWhitespace(c);
      skipToWhitespace(c);
      skipWhitespace(c);
    }
  }
}
