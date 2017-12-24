#include "printer.hpp"

namespace occa {
  namespace lang {
    int charsFromNewline(const std::string &s) {
      const char *c = s.c_str();
      const int chars = (int) s.size();
      for (int pos = (chars - 1); pos >= 0; --pos) {
        if (*c == '\n') {
          return (chars - pos - 1);
        }
      }
      return chars;
    }

    printer::printer() :
      ss(),
      outputStream(&ss),
      indent(),
      inlinedStack(),
      lastChar('\0'),
      charsFromNewline(0) {
      inlinedStack.push_back(false);
    }

    printer::printer(std::ostream &outputStream_) :
      ss(),
      outputStream(&outputStream_),
      indent(),
      inlinedStack(),
      lastChar('\0'),
      charsFromNewline(0) {
      inlinedStack.push_back(false);
    }

    int printer::size() {
      int pos = ss.tellg();
      ss.seekg(0, ss.end);
      int size = ss.tellg();
      ss.seekg(pos);
      return size;
    }

    void printer::setOutputStream(std::ostream &outputStream_) {
      outputStream = &outputStream_;
    }

    bool printer::isInlined() {
      const int count = (int) inlinedStack.size();
      return (count && inlinedStack[count - 1]);
    }

    void printer::pushInlined(const bool inlined) {
      inlinedStack.push_back(inlined);
    }

    void printer::popInlined() {
      if (inlinedStack.size()) {
        inlinedStack.pop_back();
      }
    }

    void printer::addIndentation() {
      indent += "  ";
    }

    void printer::removeIndentation() {
      const int chars = (int) indent.size();
      if (chars >= 2) {
        indent.resize(chars - 2);
      }
    }

    char printer::getLastChar() {
      return lastChar;
    }

    bool printer::lastCharNeedsWhitespace() {
      switch (lastChar) {
      case '\0':
      case '(':  case '[':
      case ' ':  case '\t': case '\r':
      case '\n': case '\v': case '\f':
        return false;
      }
      return true;
    }

    void printer::forceNextInlined() {
      lastChar = '\0';
    }

    std::string printer::indentFromNewline() {
      return std::string(charsFromNewline, ' ');
    }

    void printer::printIndentation() {
      *this << indent;
    }

    void printer::printStartIndentation() {
      if (!isInlined()) {
        *this << indent;
      } if (lastCharNeedsWhitespace()) {
        *this << ' ';
      }
    }

    void printer::printEndNewline() {
      if (!isInlined()) {
        *this << '\n';
      } else if (lastCharNeedsWhitespace()) {
        *this << ' ';
      }
    }
  }
}
