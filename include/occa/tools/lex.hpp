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

#include "occa/defines.hpp"
#include "occa/types.hpp"

namespace occa {
  namespace lex {
    extern const char whitespaceChars[];
    extern const char alphaChars[];
    extern const char numChars[];
    extern const char alphanumChars[];

    extern const char identifierStartChar[];
    extern const char identifierChars[];

    extern const char quote1Delimiters[];
    extern const char quote2Delimiters[];

    inline bool isDigit(const char c) {
      return (('0' <= c) && (c <= '9'));
    }

    inline bool isAlpha(const char c) {
      return ((('a' <= c) && (c <= 'z')) ||
              (('A' <= c) && (c <= 'Z')));
    }

    bool charIsIn(const char c, const char *delimiters);

    //---[ Skip ]-----------------------
    void skipTo(const char *&c, const char delimiter);
    void skipTo(const char *&c, const char delimiter, const char escapeChar);
    void skipTo(const char *&c, const char *delimiters);
    void skipTo(const char *&c, const char *delimiters, const char escapeChar);
    void skipTo(const char *&c, const std::string &delimiters);
    void skipTo(const char *&c, const std::string &delimiters, const char escapeChar);

    inline void skipTo(char *&c, const char delimiter) {
      skipTo((const char *&) c, delimiter);
    }

    inline void skipTo(char *&c, const char delimiter, const char escapeChar) {
      skipTo((const char *&) c, delimiter, escapeChar);
    }

    inline void skipTo(char *&c, const char *delimiters) {
      skipTo((const char *&) c, delimiters);
    }

    inline void skipTo(char *&c, const char *delimiters, const char escapeChar) {
      skipTo((const char *&) c, delimiters, escapeChar);
    }

    inline void skipTo(char *&c, const std::string &delimiters) {
      skipTo((const char *&) c, delimiters);
    }

    inline void skipTo(char *&c, const std::string &delimiters, const char escapeChar) {
      skipTo((const char *&) c, delimiters, escapeChar);
    }

    void quotedSkipTo(const char *&c, const char delimiter);
    void quotedSkipTo(const char *&c, const char *delimiters);
    void quotedSkipTo(const char *&c, const std::string &delimiters);

    inline void quotedSkipTo(char *&c, const char delimiter) {
      quotedSkipTo((const char *&) c, delimiter);
    }

    inline void quotedSkipTo(char *&c, const char *delimiters) {
      quotedSkipTo((const char *&) c, delimiters);
    }

    inline void quotedSkipTo(char *&c, const std::string &delimiters) {
      quotedSkipTo((const char *&) c, delimiters);
    }

    void skipToMatch(const char *&c, const std::string &match);
    void skipToMatch(const char *&c, const std::string &match, const char escapeChar);

    inline void skipToMatch(char *&c, const std::string &match) {
      skipToMatch((const char *&) c, match);
    }

    inline void skipToMatch(char *&c, const std::string &match, const char escapeChar) {
      skipToMatch((const char *&) c, match, escapeChar);
    }

    void skipFrom(const char *&c, const char *delimiters);
    void skipFrom(const char *&c, const char *delimiters, const char escapeChar);
    void skipFrom(const char *&c, const std::string &delimiters);
    void skipFrom(const char *&c, const std::string &delimiters, const char escapeChar);

    inline void skipFrom(char *&c, const char *delimiters) {
      skipFrom((const char *&) c, delimiters);
    }

    inline void skipFrom(char *&c, const char *delimiters, const char escapeChar) {
      skipFrom((const char *&) c, delimiters, escapeChar);
    }

    inline void skipFrom(char *&c, const std::string &delimiters) {
      skipFrom((const char *&) c, delimiters);
    }

    inline void skipFrom(char *&c, const std::string &delimiters, const char escapeChar) {
      skipFrom((const char *&) c, delimiters, escapeChar);
    }
    //==================================

    //---[ Whitespace ]-----------------
    bool isWhitespace(const char c);

    void skipWhitespace(const char *&c);
    void skipWhitespace(const char *&c, const char escapeChar);

    inline void skipWhitespace(char *&c) {
      skipWhitespace((const char *&) c);
    }

    inline void skipWhitespace(char *&c, const char escapeChar) {
      skipWhitespace((const char *&) c, escapeChar);
    }

    void skipToWhitespace(const char *&c);

    inline void skipToWhitespace(char *&c) {
      skipToWhitespace((const char *&) c);
    }

    void skipBetweenWhitespaces(const char *&c);
    void skipBetweenWhitespaces(const char *&c, const char escapeChar);

    inline void skipBetweenWhitespaces(char *&c) {
      skipBetweenWhitespaces((const char *&) c);
    }

    inline void skipBetweenWhitespaces(char *&c, const char escapeChar) {
      skipBetweenWhitespaces((const char *&) c, escapeChar);
    }

    void strip(const char *&start, const char *&end);
    void strip(const char *&start, const char *&end, const char escapeChar);

    inline void strip(char *&start, char *&end) {
      strip((const char *&) start, (const char *&) end);
    }

    inline void strip(char *&start, char *&end, const char escapeChar) {
      strip((const char *&) start, (const char *&) end, escapeChar);
    }
    //==================================
  }
}
#endif
