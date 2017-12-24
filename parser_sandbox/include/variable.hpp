#ifndef OCCA_PARSER_VARIABLE_HEADER2
#define OCCA_PARSER_VARIABLE_HEADER2

#include "type.hpp"

namespace occa {
  namespace lang {
    class variable {
    public:
      type_t *type;
      std::string name;

      variable(type_t &type_,
               const std::string &name_);

      variable& clone() const;

      void print(printer &pout) const;
    };
  }
}

#endif
