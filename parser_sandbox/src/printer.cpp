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

    printer_t::printer_t() :
      ss(),
      outputStream(&ss),
      indent(),
      inlinedStack(),
      lastChar('\0'),
      charsFromNewline(0) {
      inlinedStack.push_back(false);
    }

    printer_t::printer_t(std::ostream &outputStream_) :
      ss(),
      outputStream(&outputStream_),
      indent(),
      inlinedStack(),
      lastChar('\0'),
      charsFromNewline(0) {
      inlinedStack.push_back(false);
    }

    void printer_t::setOutputStream(std::ostream &outputStream_) {
      outputStream = &outputStream_;
    }

    bool printer_t::isInlined() {
      const int count = (int) inlinedStack.size();
      return (count && inlinedStack[count - 1]);
    }

    void printer_t::pushInlined(const bool inlined) {
      inlinedStack.push_back(inlined);
    }

    void printer_t::popInlined() {
      if (inlinedStack.size()) {
        inlinedStack.pop_back();
      }
    }

    void printer_t::addIndentation() {
      indent += "  ";
    }

    void printer_t::removeIndentation() {
      const int chars = (int) indent.size();
      if (chars >= 2) {
        indent.resize(chars - 2);
      }
    }

    bool printer_t::lastCharNeedsWhitespace() {
      switch (lastChar) {
      case '\0':
      case '(':  case '[':
      case ' ':  case '\t': case '\r':
      case '\n': case '\v': case '\f':
        return false;
      }
      return true;
    }

    void printer_t::forceNextInlined() {
      lastChar = '\0';
    }

    std::string printer_t::indentFromNewline() {
      return std::string(charsFromNewline, ' ');
    }

    void printer_t::printIndentation() {
      *this << indent;
    }

    void printer_t::printStartIndentation() {
      if (!isInlined()) {
        *this << indent;
      } if (lastCharNeedsWhitespace()) {
        *this << ' ';
      }
    }

    void printer_t::printEndNewline() {
      if (!isInlined()) {
        *this << '\n';
      } else if (lastCharNeedsWhitespace()) {
        *this << ' ';
      }
    }
  }
}
