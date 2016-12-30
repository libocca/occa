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

#ifndef OCCA_TOOLS_LEX_HEADER
#define OCCA_TOOLS_LEX_HEADER

#include <iostream>
#include <iomanip>
#include <sstream>

#include "occa/defines.hpp"
#include "occa/types.hpp"

namespace occa {
  namespace lex {
    extern const char whitespaceChars[];

    void skipTo(const char *&c, const char delimiter);
    void skipTo(const char *&c, const char delimiter, const char escapeChar);

    void skipTo(const char *&c, const std::string &match);
    void skipTo(const char *&c, const std::string &match, const char escapeChar);

    void skipToDelimiter(const char *&c, const std::string &delimiters);
    void skipToDelimiter(const char *&c, const std::string &delimiters, const char escapeChar);

    bool charIsIn(const char c, const char *delimiters);

    bool isWhitespace(const char c);
    void skipWhitespace(const char *&c);
    void skipToWhitespace(const char *&c);
    void skipBetweenWhitespaces(const char *&c);
  }
}
#endif
