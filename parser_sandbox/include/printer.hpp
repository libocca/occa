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
#ifndef OCCA_PARSER_TOOLS_HEADER2
#define OCCA_PARSER_TOOLS_HEADER2

#include <sstream>
#include <iostream>
#include <vector>

namespace occa {
  namespace lang {
    int charsFromNewline(const std::string &s);

    class printer {
    private:
      std::stringstream ss;
      std::ostream *outputStream;

      std::string indent;
      std::vector<int> inlinedStack;

      // Metadata
      char lastChar;
      int charsFromNewline;

    public:
      printer();
      printer(std::ostream &outputStream_);

      void setOutputStream(std::ostream &outputStream_);

      int size();

      bool isInlined();
      void pushInlined(const bool inlined);
      void popInlined();

      void addIndentation();
      void removeIndentation();

      char getLastChar();
      bool lastCharNeedsWhitespace();
      void forceNextInlined();

      std::string indentFromNewline();

      void printIndentation();
      void printStartIndentation();
      void printEndNewline();

      template <class TM>
      printer& operator << (const TM &t) {
        ss << t;
        const std::string str = ss.str();
        const char *c_str = str.c_str();
        const int chars = (int) str.size();
        if (chars) {
          ss.str("");
          lastChar = c_str[chars - 1];
          for (int i = 0; i < chars; ++i) {
            if (c_str[i] != '\n') {
              ++charsFromNewline;
            } else {
              charsFromNewline = 0;
            }
          }
          *outputStream << str;
        }
        return *this;
      }
    };
  }
}

#endif
