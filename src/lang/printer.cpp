#include <occa/internal/lang/printer.hpp>

namespace occa {
  namespace lang {
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

      for (int i = 0; i < lastCharsBufferSize; ++i) {
        lastChars[i] = '\0';
      }
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

    int printer::indentationSize() {
      return (int) indent.size();
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
      return lastChars[0];
    }

    bool printer::lastCharNeedsWhitespace() {
      switch (lastChars[0]) {
      case '\0':
      case '(':  case '[':
      case ' ':  case '\t': case '\r':
      case '\n': case '\v': case '\f':
        return false;
      }
      return true;
    }

    void printer::forceNextInlined() {
      lastChars[0] = '\0';
    }

    int printer::cursorPosition() {
      return charsFromNewline;
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

    void printer::printSpace() {
      if (lastChars[0] != ' ') {
        *this << ' ';
      }
    }

    void printer::printNewline() {
      if (lastChars[0] != '\n') {
        *this << '\n';
      }
    }

    void printer::printNewlines(const int count) {
      const int newlines = (
        count <= lastCharsBufferSize
        ? count
        : lastCharsBufferSize
      );

      // Avoid printing newlines if we already have some
      bool needsNewline = false;
      for (int i = 0; i < newlines; ++i) {
        needsNewline = needsNewline || lastChars[i] != '\n';
        if (needsNewline) {
          *this << '\n';
        }
      }
    }

    void printer::printEndNewline() {
      if (!isInlined()) {
        if (lastChars[0] != '\n') {
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
