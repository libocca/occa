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

#ifndef OCCA_TOOLS_LEX_HEADER
#define OCCA_TOOLS_LEX_HEADER

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
