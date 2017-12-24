#include "variable.hpp"
#include "statement.hpp"

namespace occa {
  namespace lang {
    variable::variable(type_t &type_,
                       const std::string &name_) :
      type(&type_),
      name(name_) {}

    variable& variable::clone() const {
      return *(new variable(type->clone(), name));
    }

    void variable::print(printer &pout) const {
      pout << name;
    }
  }
}
