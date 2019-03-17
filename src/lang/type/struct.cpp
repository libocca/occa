#include <occa/lang/type/struct.hpp>

namespace occa {
  namespace lang {
    struct_t::struct_t() :
      structure_t("") {}

    int struct_t::type() const {
      return typeType::struct_;
    }

    type_t& struct_t::clone() const {
      return *(new struct_t());
    }

    dtype_t struct_t::dtype() const {
      return dtype::byte;
    }

    void struct_t::printDeclaration(printer &pout) const {
    }
  }
}
