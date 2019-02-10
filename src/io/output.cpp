#include <occa/io/output.hpp>

namespace occa {
  namespace io {
    output::output(std::ostream &out_) :
      out(out_),
      customOut(NULL) {}

    void output::setOutputFunction(outputFunction_t customOut_) {
      customOut = customOut_;
    }

    template <>
    output& output::operator << (const std::string &str) {
      if (!customOut) {
        out << str;
      } else {
        customOut(str.c_str());
      }
      return *this;
    }

    template <>
    output& output::operator << (char * const &c) {
      if (!customOut) {
        out << c;
      } else {
        customOut(c);
      }
      return *this;
    }

    template <>
    output& output::operator << (const char &c) {
      if (!customOut) {
        out << c;
      } else {
        char str[2] = { c, '\0' };
        customOut(str);
      }
      return *this;
    }

    output stdout(std::cout);
    output stderr(std::cerr);
  }
}
