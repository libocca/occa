#include <occa/lang/printer.hpp>

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
      out(NULL) {
      clear();
    }

    printer::printer(io::output &out_) :
      ss(),
      out(&out_) {
      clear();
    }

    void printer::setOutput(io::output &out_) {
      out = &out_;
    }

    int printer::size() {
      int pos = ss.tellg();
      ss.seekg(0, ss.end);
      int size = ss.tellg();
      ss.seekg(pos);
      return size;
    }

    std::string printer::str() {
      return ss.str();
    }

    void printer::clear() {
      ss.str("");
      indent = "";

      inlinedStack.clear();
      inlinedStack.push_back(false);

      lastChar = '\0';
      charsFromNewline = 0;
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
      } else if (lastCharNeedsWhitespace()) {
        *this << ' ';
      }
    }

    void printer::printNewline() {
      if (lastChar != '\n') {
        *this << '\n';
      }
    }

    void printer::printEndNewline() {
      if (!isInlined()) {
        if (lastChar != '\n') {
          *this << '\n';
        }
      } else if (lastCharNeedsWhitespace()) {
        *this << ' ';
      }
    }

    printer& operator << (printer &pout,
                          const std::string &str) {
      pout.print(str);
      return pout;
    }

    printer& operator << (printer &pout,
                          const char c) {
      pout.print(c);
      return pout;
    }
  }
}
