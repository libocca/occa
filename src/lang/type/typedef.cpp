#include <occa/lang/type/typedef.hpp>

namespace occa {
  namespace lang {
    typedef_t::typedef_t(const vartype_t &baseType_) :
      type_t(),
      baseType(baseType_) {}

    typedef_t::typedef_t(const vartype_t &baseType_,
                         identifierToken &source_) :
      type_t(source_),
      baseType(baseType_) {}

    typedef_t::typedef_t(const typedef_t &other) :
      type_t(other),
      baseType(other.baseType) {}

    int typedef_t::type() const {
      return typeType::typedef_;
    }

    type_t& typedef_t::clone() const {
      return *(new typedef_t(*this));
    }

    bool typedef_t::isPointerType() const {
      return baseType.isPointerType();
    }

    dtype_t typedef_t::dtype() const {
      return baseType.dtype();
    }

    bool typedef_t::equals(const type_t &other) const {
      return (baseType == other.to<typedef_t>().baseType);
    }

    void typedef_t::printDeclaration(printer &pout) const {
      pout << "typedef ";
      baseType.printDeclaration(pout, name());
    }
  }
}
