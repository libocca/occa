#include <occa/internal/lang/type/union.hpp>

namespace occa {
  namespace lang {
    union_t::union_t() :
      structure_t("") {}

    int union_t::type() const {
      return typeType::union_;
    }

    type_t& union_t::clone() const {
      return *(new union_t());
    }

    dtype_t union_t::dtype() const {
      return dtype::byte;
    }

    void union_t::printDeclaration(printer &pout) const {
    }
  }
}
