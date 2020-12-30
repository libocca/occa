#include <occa/internal/io/output.hpp>

namespace occa {
  namespace io {
    output::output(std::ostream &out_) :
      out(out_),
      overrideOut(NULL) {}

    void output::setOverride(outputFunction_t overrideOut_) {
      overrideOut = overrideOut_;
    }

    template <>
    output& output::operator << (const std::string &str) {
      if (!overrideOut) {
        out << str;
      } else {
        overrideOut(str.c_str());
      }
      return *this;
    }

    template <>
    output& output::operator << (char * const &c) {
      if (!overrideOut) {
        out << c;
      } else {
        overrideOut(c);
      }
      return *this;
    }

    template <>
    output& output::operator << (const char &c) {
      if (!overrideOut) {
        out << c;
      } else {
        char str[2] = { c, '\0' };
        overrideOut(str);
      }
      return *this;
    }

    output stdout(std::cout);
    output stderr(std::cerr);
  }
}
