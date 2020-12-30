#include <occa/internal/lang/type/primitive.hpp>

namespace occa {
  namespace lang {
    primitive_t::primitive_t(const std::string &name_) :
      type_t(name_),
      pname(name_),
      dtype_(NULL) {}

    const std::string& primitive_t::name() const {
      return pname;
    }

    bool primitive_t::isNamed() const {
      return true;
    }

    int primitive_t::type() const {
      return typeType::primitive;
    }

    type_t& primitive_t::clone() const {
      return *(const_cast<primitive_t*>(this));
    }

    dtype_t primitive_t::dtype() const {
      if (!dtype_) {
        dtype_ = &(dtype_t::getBuiltin(pname));
      }
      return *dtype_;
    }

    void primitive_t::printDeclaration(printer &pout) const {
      pout << name();
    }
  }
}
