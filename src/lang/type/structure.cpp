#include <occa/internal/lang/type/structure.hpp>

namespace occa {
  namespace lang {
    structure_t::structure_t(const std::string &name_) :
      type_t(name_),
      body(NULL) {}
  }
}
