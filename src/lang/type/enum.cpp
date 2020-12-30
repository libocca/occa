#include <occa/internal/lang/type/enum.hpp>

namespace occa {
  namespace lang {
    enum_t::enum_t() :
      structure_t("") {}

    int enum_t::type() const {
      return typeType::enum_;
    }

    type_t& enum_t::clone() const {
      return *(new enum_t());
    }

    dtype_t enum_t::dtype() const {
      return dtype::byte;
    }

    void enum_t::printDeclaration(printer &pout) const {
    }
  }
}
