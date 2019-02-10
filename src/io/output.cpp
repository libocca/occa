#include <occa/io/output.hpp>

namespace occa {
  namespace io {
    output_t::output_t(std::ostream &out_) :
      out(out_),
      customOut(NULL) {}

    void output_t::setOutputFunction(outputFunction_t customOut_) {
      customOut = customOut_;
    }

    output_t::operator std::ostream& () {
      return out;
    }

    template <>
    output_t& output_t::operator << (const std::string &t) {
      if (!customOut) {
        out << t;
      } else {
        out << t;
        customOut(t.c_str());
      }
      return *this;
    }

    output_t stdout(std::cout);
    output_t stderr(std::cerr);
  }
}
