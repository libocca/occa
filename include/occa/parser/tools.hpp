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

#ifndef OCCA_PARSER_TOOLS_HEADER
#define OCCA_PARSER_TOOLS_HEADER

#include <iomanip>

#include "occa/defines.hpp"
#include "occa/parser/defines.hpp"

namespace occa {
  //---[ Helper Functions ]-----------------------
  bool stringsAreEqual(const char *cStart, const size_t chars,
                       const char *c2);

  bool charIsIn(const char c, const char *delimiters);
  bool charIsIn2(const char *c, const char *delimiters);
  bool charIsIn3(const char *c, const char *delimiters);

  char upChar(const char c);
  char downChar(const char c);

  std::string upString(const char *c, const int chars);
  std::string upString(const std::string &s);

  bool upStringCheck(const std::string &a,
                     const std::string &b);

  inline char back(std::string &s) {
    return s[s.size() - 1];
  }

  bool isWhitespace(const char c);
  void skipWhitespace(const char *&c);
  void skipToWhitespace(const char *&c);

  int charsBeforeNewline(const std::string &str);

  bool isAString(const char *c);
  bool isAnInt(const char *c);
  bool isAFloat(const char *c);
  bool isANumber(const char *c);

  bool isAString(const std::string &str);
  bool isAnInt(const std::string &str);
  bool isAFloat(const std::string &str);
  bool isANumber(const std::string &str);

  inline bool isADigit(const char c) {
    return (('0' <= c) && (c <= '9'));
  }

  inline bool isAlpha(const char c) {
    return ((('a' <= c) && (c <= 'z')) ||
            (('A' <= c) && (c <= 'Z')));
  }

  void skipInt(const char *&c);

  void skipNumber(const char *&c,
                  const int parsingLanguage_ = parserInfo::parsingC);
  void skipFortranNumber(const char *&c);

  void skipString(const char *&c,
                  const int parsingLanguage_ = parserInfo::parsingC);

  char isAWordDelimiter(const char *c,
                        const int parsingLanguage_ = parserInfo::parsingC);
  char isAFortranWordDelimiter(const char *c);

  int skipWord(const char *&c,
               const int parsingLanguage_ = parserInfo::parsingC);

  bool isAnUpdateOperator(const std::string &s,
                          const int parsingLanguage_ = parserInfo::parsingC);
  bool isAnAssOperator(const std::string &s,
                       const int parsingLanguage_ = parserInfo::parsingC); // hehe
  bool isAnInequalityOperator(const std::string &s,
                              const int parsingLanguage_ = parserInfo::parsingC);

  const char* readLine(const char *c,
                       const int parsingLanguage_ = parserInfo::parsingC);
  const char* readFortranLine(const char *c);

  std::string compressWhitespace(const std::string &str);

  std::string compressAllWhitespace(const char *c, const size_t chars,
                                    const int parsingLanguage_ = parserInfo::parsingC);
  void compressAllWhitespace(std::string &str,
                             const int parsingLanguage_ = parserInfo::parsingC);

  int stripComments(std::string &line,
                    const int parsingLanguage_ = parserInfo::parsingC);
  int stripFortranComments(std::string &line);

  bool charStartsSection(const char c);
  bool charEndsSection(const char c);

  bool startsSection(const std::string &str);
  bool endsSection(const std::string &str);

  char segmentPair(const char c);
  char segmentPair(const std::string &str);
  void skipPair(const char *&c);

  int countDelimiters(const char *c, const char delimiter);

  void skipTo(const char *&c, const char delimiter, const char escapeChar = 0);
  void skipTo(const char *&c, std::string delimiters, const char escapeChar = 0);
  void skipToWord(const char *&c, std::string word);

  std::string findFileInPath(const std::string &filename);

  template <class TM>
  inline void swapValues(TM &a, TM &b) {
    TM a_ = a;
    a = b;
    b = a_;
  }
  //==============================================

  std::string getBits(const info_t value);
}

#endif
