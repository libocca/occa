#ifndef OCCA_PARSER_VARIABLE_HEADER2
#define OCCA_PARSER_VARIABLE_HEADER2

#include "type.hpp"

namespace occa {
  namespace lang {
    class variable_t {
    public:
      type_t *type;
      std::string name;

      variable_t(type_t &type_,
                 const std::string &name_);

      variable_t& clone() const;

      void print(printer_t &pout) const;
    };
  }
}

#endif
