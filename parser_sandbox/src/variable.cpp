#include "variable.hpp"
#include "statement.hpp"

namespace occa {
  namespace lang {
    variable_t::variable_t(type_t &type_,
                           const std::string &name_) :
      type(&type_),
      name(name_) {}

    variable_t& variable_t::clone() const {
      return *(new variable_t(type->clone(), name));
    }

    void variable_t::print(printer_t &pout) const {
      pout << name;
    }
  }
}
