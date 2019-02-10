#ifndef OCCA_IO_OUTPUT_HEADER
#define OCCA_IO_OUTPUT_HEADER

#include <iostream>
#include <sstream>

namespace occa {
  namespace io {
    typedef void (*outputFunction_t)(const char *str);

    class output_t {
    private:
      std::ostream &out;
      std::stringstream ss;
      outputFunction_t customOut;

    public:
      output_t(std::ostream &out_);

      void setOutputFunction(outputFunction_t customOut_);

      operator std::ostream& ();

      template <class TM>
      output_t& operator << (const TM &t);
    };

    template <>
    output_t& output_t::operator << (const std::string &t);

    extern output_t stdout;
    extern output_t stderr;
  }
}

#include "output.tpp"

#endif
