#ifndef OCCA_INTERNAL_LANG_PRINTER_HEADER
#define OCCA_INTERNAL_LANG_PRINTER_HEADER

#include <sstream>
#include <iostream>
#include <vector>

#include <occa/internal/io/output.hpp>

namespace occa {
  namespace lang {
    class printer {
    private:
      std::stringstream ss;
      io::output *out;

      std::string indent;
      std::vector<int> inlinedStack;

      // Metadata
      static const int lastCharsBufferSize = 10;
      // Keeps the last character in the [0] position
      char lastChars[lastCharsBufferSize];
      int charsFromNewline;

    public:
      printer();
      printer(io::output &out_);

      void setOutput(io::output &out_);

      int size();

      std::string str();
      void clear();

      bool isInlined();
      void pushInlined(const bool inlined);
      void popInlined();

      int indentationSize();
      void addIndentation();
      void removeIndentation();

      char getLastChar();
      bool lastCharNeedsWhitespace();
      void forceNextInlined();

      int cursorPosition();
      std::string indentFromNewline();

      void printIndentation();
      void printStartIndentation();
      void printSpace();
      void printNewline();
      void printNewlines(const int count);
      void printEndNewline();

      template <class TM>
      void print(const TM &t) {
        ss << t;
        const std::string str = ss.str();
        const int chars = (int) str.size();
        if (!chars) {
          return;
        }

        int scanStart;
        if (out) {
          // Clear buffer
          ss.str("");
          scanStart = 0;
        } else {
          // We don't clear the buffer so no need to re-read past characters
          scanStart = charsFromNewline;
        }

        const char *c_str = str.c_str();
        for (int i = scanStart; i < chars; ++i) {
          if (c_str[i] != '\n') {
            ++charsFromNewline;
          } else {
            charsFromNewline = 0;
          }
        }

        const int replacedLastChars = (
          chars > lastCharsBufferSize
          ? lastCharsBufferSize
          : chars
        );

        // Slide remaining characters
        for (int i = replacedLastChars; i < lastCharsBufferSize; ++i) {
          lastChars[i] = lastChars[i - replacedLastChars];
        }
        // Replace with new last characters
        for (int i = 0; i < replacedLastChars; ++i) {
          lastChars[i] = c_str[chars - 1 - i];
        }

        if (out) {
          // Print to buffer
          *out << str;
        }
      }
    };

    printer& operator << (printer &pout,
                          const std::string &str);

    printer& operator << (printer &pout,
                          const char c);
  }
}

#endif
