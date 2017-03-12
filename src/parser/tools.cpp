/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
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

#include "occa/parser/preprocessor.hpp"
#include "occa/parser/tools.hpp"
#include "occa/tools/env.hpp"
#include "occa/tools/io.hpp"
#include "occa/tools/sys.hpp"

namespace occa {
  //---[ Helper Functions ]-----------------------
  bool stringsAreEqual(const char *cStart, const size_t chars,
                       const char *c2) {
    for (size_t c = 0; c < chars; ++c) {
      if (cStart[c] != c2[c])
        return false;

      if ((cStart[c] == '\0') || (c2[c] == '\0'))
        return false;
    }

    return true;
  }

  bool charIsIn(const char c, const char *delimiters) {
    while ((*delimiters) != '\0')
      if (c == *(delimiters++))
        return true;

    return false;
  }

  bool charIsIn2(const char *c, const char *delimiters) {
    if ((c[0] == '\0') || (c[1] == '\0'))
      return false;

    const char c0 = c[0];
    const char c1 = c[1];

    while ((*delimiters) != '\0') {
      if ((c0 == delimiters[0]) && (c1 == delimiters[1]))
        return true;

      delimiters += 2;
    }

    return false;
  }

  bool charIsIn3(const char *c, const char *delimiters) {
    if ((c[0] == '\0') || (c[1] == '\0') || (c[2] == '\0'))
      return false;

    const char c0 = c[0];
    const char c1 = c[1];
    const char c2 = c[2];

    while ((*delimiters) != '\0') {
      if ((c0 == delimiters[0]) && (c1 == delimiters[1]) && (c2 == delimiters[2]))
        return true;

      delimiters += 3;
    }

    return false;
  }

  char upChar(const char c) {
    if (('a' <= c) && (c <= 'z'))
      return ((c + 'A') - 'a');

    return c;
  }

  char downChar(const char c) {
    if (('A' <= c) && (c <= 'Z'))
      return ((c + 'a') - 'A');

    return c;
  }

  std::string upString(const char *c, const int chars) {
    std::string ret(c, chars);

    for (int i = 0; i < chars; ++i)
      ret[i] = upChar(ret[i]);

    return ret;
  }

  std::string upString(const std::string &s) {
    return upString(s.c_str(), s.size());
  }

  bool upStringCheck(const std::string &a,
                     const std::string &b) {
    const int aSize = a.size();
    const int bSize = b.size();

    if (aSize != bSize)
      return false;

    for (int i = 0; i < aSize; ++i) {
      if (upChar(a[i]) != upChar(b[i]))
        return false;
    }

    return true;
  }

  bool isWhitespace(const char c) {
    return charIsIn(c, parserNS::whitespace);
  }

  void skipWhitespace(const char *&c) {
    while (charIsIn(*c, parserNS::whitespace) && (*c != '\0'))
      ++c;
  }

  void skipToWhitespace(const char *&c) {
    while (!charIsIn(*c, parserNS::whitespace) && (*c != '\0'))
      ++c;
  }

  int charsBeforeNewline(const std::string &str) {
    const int chars = (int) str.size();
    const char *c0  = str.c_str();
    const char *c;

    for (c = (c0 + chars - 1); c0 <= c; --c) {
      if (*c == '\n')
        break;
    }

    return (chars - (c - c0) - 1);
  }

  bool isAString(const char *c) {
    return ((*c == '\'') || (*c == '"'));
  }

  bool isAnInt(const char *c) {
    parserNS::typeHolder th;
    th.load(c);

    return (th.isAnInt());
  }

  bool isAFloat(const char *c) {
    parserNS::typeHolder th;
    th.load(c);

    return (th.isAFloat());
  }

  bool isANumber(const char *c) {
    parserNS::typeHolder th;
    th.load(c);

    return (th.type != parserNS::noType);
  }

  bool isAString(const std::string &str) {
    return isAString(str.c_str());
  }

  bool isAnInt(const std::string &str) {
    return isAnInt(str.c_str());
  }

  bool isAFloat(const std::string &str) {
    return isAFloat(str.c_str());
  }

  bool isANumber(const std::string &str) {
    return isANumber(str.c_str());
  }

  void skipInt(const char *&c) {
    parserNS::typeHolder th;
    th.load(c);
  }

  void skipNumber(const char *&c, const int parsingLanguage_) {
    parserNS::typeHolder th;
    th.load(c);
  }

  void skipFortranNumber(const char *&c) {
    if ((*c == '+') || (*c == '-'))
      ++c;

    skipInt(c);

    if (*c == '.') {
      ++c;

      skipInt(c);
    }

    const char nextChar = upChar(*c);

    if ((nextChar == 'D') ||
       (nextChar == 'E')) {
      ++c;

      if ((*c == '+') || (*c == '-'))
        ++c;

      skipInt(c);
    }

    if (*c == '_')
      c += 2;
  }

  void skipString(const char *&c, const int parsingLanguage_) {
    if (!isAString(c))
      return;

    const char nl = ((parsingLanguage_ & parserInfo::parsingC) ? '\\' : '&');

    const char match = *(c++);

    while (*c != '\0') {
      if (*c == nl)
        ++c;
      else if (*c == match) {
        ++c;
        return;
      }

      ++c;
    }
  }

  char isAWordDelimiter(const char *c, const int parsingLanguage_) {
    if (!(parsingLanguage_ & parserInfo::parsingC))
      return isAFortranWordDelimiter(c);

    if (charIsIn3(c, parserNS::cWordDelimiter3))
      return 3;

    if (charIsIn2(c, parserNS::cWordDelimiter2))
      return 2;

    if (charIsIn(c[0], parserNS::cWordDelimiter))
      return 1;

    return 0;
  }

  char isAFortranWordDelimiter(const char *c) {
    if (charIsIn2(c, parserNS::fortranWordDelimiter2)) {
      return 2;
    }
    else if (charIsIn(c[0], parserNS::fortranWordDelimiter)) {
      if (c[0] == '.') {
        const char *c2 = (c + 1);

        while (*c2 != '.')
          ++c2;

        return (c2 - c + 1);
      }

      return 1;
    }

    return 0;
  }

  int skipWord(const char *&c, const int parsingLanguage_) {
    while (!charIsIn(*c, parserNS::whitespace) && (*c != '\0')) {
      const int delimiterChars = isAWordDelimiter(c, parsingLanguage_);

      if (delimiterChars == 0)
        ++c;
      else
        return delimiterChars;
    }

    return 0;
  }

  bool isAnUpdateOperator(const std::string &s,
                          const int parsingLanguage_) {
    if (isAnAssOperator(s, parsingLanguage_))
      return true;

    return (s == "++" || s == "--");
  }

  bool isAnAssOperator(const std::string &s,
                       const int parsingLanguage_) { // hehe
    const size_t chars = s.size();
    const char *c      = s.c_str();

    if ((chars < 1) ||
       (3 < chars) ||          // Not in range
       (c[chars - 1] != '=')) { // Not an assignment operator

      return false;
    }

    if (chars == 1) {      // =
      return true;
    }
    else if (chars == 2) { // +=, -=, *=, /=, %=, &=, ^=, |=
      if ((c[0] == '+') ||
         (c[0] == '-') ||
         (c[0] == '*') ||
         (c[0] == '/') ||
         (c[0] == '%') ||
         (c[0] == '&') ||
         (c[0] == '^') ||
         (c[0] == '|')) {

        return true;
      }
    }
    else {                // <<=, >>=
      if (((c[0] == '<') && (c[1] == '<')) ||
         ((c[0] == '>') && (c[1] == '>'))) {

        return true;
      }
    }

    return false;
  }

  bool isAnInequalityOperator(const std::string &s, const int parsingLanguage_) {
    const size_t chars = s.size();
    const char *c      = s.c_str();

    const bool hasEQ = ((c[0] == '<') || (c[0] == '>'));

    if (!hasEQ)
      return false;

    if (chars == 1)
      return hasEQ;
    else if (chars == 2)
      return (c[1] == '=');

    return false;
  }

  const char* readLine(const char *c, const int parsingLanguage_) {
    if (!(parsingLanguage_ & parserInfo::parsingC))
      return readFortranLine(c);

    bool breakNextLine = true;

    while (*c != '\0') {
      if (*c == '\0')
        break;

      if (*c == '\n') {
        if (breakNextLine)
          break;

        breakNextLine = false;
      }
      // Append next line
      else if ((c[0] == '\\') && isWhitespace(c[1])) {
        breakNextLine = true;
        ++c;
      }
      else if (c[0] == '/') {
        if (c[1] == '/') {
          while ((*c != '\n') && (*c != '\0'))
            ++c;

          return c;
        }
        else if (c[1] == '*') {
          c += 2;

          while ( !((c[0] == '*') && (c[1] == '/')) &&
                 (*c != '\0') )
            ++c;

          if (*c == '*')
            c += 2;

          return c;
        }
      }

      ++c;
    }

    return ((c[0] != '\0') ? (c + 1) : c);
  }

  const char* readFortranLine(const char *c) {
    bool breakNextLine = true;

    // Starting with [c] means line is a comment
    if (*c == 'c') {
      while ((*c != '\n') &&
            (*c != '\0')) {

        ++c;
      }

      return c;
    }

    while (*c != '\0') {
      if (*c == '\0')
        break;

      if (*c == '\n') {
        if (breakNextLine)
          break;

        breakNextLine = false;
      }
      // Append next line
      else if ((c[0] == '&') && isWhitespace(c[1])) {
        breakNextLine = true;
        ++c;
      }
      else if (c[0] == '!') {
        while ((*c != '\n') && (*c != '\0'))
          ++c;

        return c;
      }

      ++c;
    }

    return ((c[0] != '\0') ? (c + 1) : c);
  }

  std::string compressWhitespace(const std::string &str) {
    std::string ret = str;

    const char *c = str.c_str();
    size_t pos = 0;

    while (*c != '\0') {
      if (isWhitespace(*c)) {
        ret[pos++] = ' ';

        skipWhitespace(c);
      }
      else
        ret[pos++] = *(c++);
    }

    ret.resize(pos);

    return ret;
  }

  std::string compressAllWhitespace(const char *c,
                                    const size_t chars,
                                    const int parsingLanguage_) {
    if (chars == 0)
      return "";

    const char nl = ((parsingLanguage_ & parserInfo::parsingC) ? '\\' : '&');

    const char *cLeft  = c;
    const char *cRight = c + (chars - 1);

    while (charIsIn(*cLeft , parserNS::whitespace) && (cLeft <= cRight)) ++cLeft;
    while (charIsIn(*cRight, parserNS::whitespace) && (cRight > cLeft)) --cRight;

    if (cLeft > cRight) {
      return "";
    }
    std::string ret = "";

    const char *cMid = cLeft;

    while (cMid < cRight) {
      if ((cMid[0] == nl) && isWhitespace(cMid[1])) {
        ret += compressAllWhitespace(cLeft, cMid - cLeft);
        ret += ' ';

        ++cMid;

        cLeft = (cMid + 1);
      }

      ++cMid;

      if ((cMid >= cRight) && ret.size())
        ret += compressAllWhitespace(cLeft, (cMid - cLeft + 1));
    }

    if (ret.size() == 0) {
      return compressWhitespace( std::string(cLeft, (cRight - cLeft + 1)) );
    }
    return compressWhitespace(ret);
  }

  void compressAllWhitespace(std::string &str,
                       const int parsingLanguage_) {
    str = compressAllWhitespace(str.c_str(), str.size());
  }

  int stripComments(std::string &line, const int parsingLanguage_) {
    if (!(parsingLanguage_ & parserInfo::parsingC))
      return stripFortranComments(line);

    std::string line2 = line;
    line = "";

    const char *cLeft  = line2.c_str();
    const char *cRight = cLeft;

    info_t status = parserNS::readingCode;

    while (*cRight != '\0') {
      if ((*cRight == '\0') || (*cRight == '\n'))
        break;

      skipString(cRight, parsingLanguage_);

      if ((cRight[0] == '/') && (cRight[1] == '/')) {
        if ( !(status == parserNS::insideCommentBlock) ) {
          line += std::string(cLeft, cRight - cLeft);
          return parserNS::readingCode;
        }
      }
      else if ((cRight[0] == '/') && (cRight[1] == '*')) {
        if ( !(status == parserNS::insideCommentBlock) ) {
          line += std::string(cLeft, cRight - cLeft);
          status = parserNS::insideCommentBlock;
        }

        cLeft = cRight + 2;
      }
      else if ((cRight[0] == '*') && (cRight[1] == '/')) {
        if (status == parserNS::insideCommentBlock)
          status = parserNS::readingCode;
        else
          status = parserNS::finishedCommentBlock;

        cLeft = cRight + 2;
      }

      ++cRight;
    }

    if (cLeft < cRight)
      line += std::string(cLeft, cRight - cLeft);

    return status;
  }

  int stripFortranComments(std::string &line) {
    std::string line2  = line;
    line = "";

    const char *cLeft  = line2.c_str();
    const char *cRight = cLeft;

    int status = parserNS::readingCode;

    while (*cRight != '\0') {
      if ((*cRight == '\0') || (*cRight == '\n'))
        break;

      if (*cRight == '!') {
        line += std::string(cLeft, cRight - cLeft);
        return parserNS::readingCode;
      }

      ++cRight;
    }

    if (cLeft != cRight)
      line += std::string(cLeft, cRight - cLeft);

    return status;
  }

  bool charStartsSection(const char c) {
    return ((c == '(') ||
            (c == '[') ||
            (c == '{'));
  }

  bool charEndsSection(const char c) {
    return ((c == ')') ||
            (c == ']') ||
            (c == '}'));
  }

  bool startsSection(const std::string &str) {
    if (str.size() == 1)
      return charStartsSection(str[0]);

    return false;
  }

  bool endsSection(const std::string &str) {
    if (str.size() == 1)
      return charEndsSection(str[0]);

    return false;
  }

  char segmentPair(const char c) {
    return ((')' * (c == '(')) +
            (']' * (c == '[')) +
            ('}' * (c == '{')) +
            ('(' * (c == ')')) +
            ('[' * (c == ']')) +
            ('{' * (c == '}')));
  }

  char segmentPair(const std::string &str) {
    if (str.size() == 1)
      return segmentPair(str[0]);

    return '\0';
  }

  void skipPair(const char *&c) {
    if (*c == '\0')
      return;

    const char pair = segmentPair(*c);

    if (pair == 0)
      return;

    ++c;

    while ((*c != '\0') &&
          (*c != pair)) {
      if (segmentPair(*c))
        skipPair(c);
      else
        ++c;
    }

    if (*c != '\0')
      ++c;
  }

  int countDelimiters(const char *c, const char delimiter) {
    int count = 0;

    while (*c != '\0') {
      if (*c == delimiter)
        ++count;

      ++c;
    }

    return count;
  }

  void skipTo(const char *&c, const char delimiter, const char escapeChar) {
    while (*c != '\0') {
      if (!(escapeChar && (*c == escapeChar)) &&
          (*c == delimiter)) {
        return;
      }
      ++c;
    }
  }

  void skipTo(const char *&c, std::string delimiters, const char escapeChar) {
    const size_t chars = delimiters.size();
    const char *d      = delimiters.c_str();

    while (*c != '\0') {
      if (!(escapeChar && (*c == escapeChar))) {
        for (size_t i = 0; i < chars; ++i) {
          if (*c == d[i]) {
            return;
          }
        }
      }
      ++c;
    }
  }

  void skipToWord(const char *&c, std::string word) {
    const size_t chars = word.size();
    const char *d      = word.c_str();

    while (*c != '\0') {
      size_t i;

      for (i = 0; i < chars; ++i) {
        if ((c[i] == '\0') ||
           (c[i] == d[i])) {
          break;
        }
      }

      if (i == chars) {
        return;
      }

      ++c;
    }
  }

  std::string findFileInPath(const std::string &filename) {
    const char *c0 = env::PATH.c_str();
    const char *c1 = c0;

    while (*c1 != '\0') {
      while ((*c1 != ':') && (*c1 != '\0'))
        ++c1;

      std::string fullFilename = std::string(c0, c1 - c0) + filename;

      if (sys::fileExists(fullFilename)) {
        return fullFilename;
      }
      if (*c1 != '\0') {
        c0 = ++c1;
      }
    }

    return "";
  }
  //==============================================

  std::string getBits(const info_t value) {
    if (value == 0)
      return "0";

    std::stringstream ret;

    bool printedSomething = false;

   for (info_t i = 0; i < (8*sizeof(info_t)); ++i) {
     if (value & (((info_t) 1) << i)) {
        if (printedSomething)
          ret << ',';

        ret << i;

        printedSomething = true;
      }
    }

    return ret.str();
  }
  //==============================================
}
