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

#include <occa/tools/lex.hpp>

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
