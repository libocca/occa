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
