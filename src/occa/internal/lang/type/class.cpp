#include <occa/internal/lang/type/class.hpp>

namespace occa {
  namespace lang {
    class_t::class_t() :
      structure_t("") {}

    int class_t::type() const {
      return typeType::class_;
    }

    type_t& class_t::clone() const {
      return *(new class_t());
    }

    dtype_t class_t::dtype() const {
      return dtype::byte;
    }

    void class_t::printDeclaration(printer &pout) const {
    }
  }
}
