#ifndef OCCA_INTERNAL_LANG_TYPE_TYPEDEF_HEADER
#define OCCA_INTERNAL_LANG_TYPE_TYPEDEF_HEADER

#include <occa/internal/lang/type/type.hpp>
#include <occa/internal/lang/type/vartype.hpp>

namespace occa {
  namespace lang {
    class typedef_t : public type_t {
    public:
      vartype_t baseType;
      bool declaredBaseType;

      typedef_t(const vartype_t &baseType_);

      typedef_t(const vartype_t &baseType_,
                identifierToken &source_);

      typedef_t(const typedef_t &other);

      ~typedef_t();

      virtual int type() const;
      virtual type_t& clone() const;

      virtual bool isPointerType() const;

      virtual dtype_t dtype() const;

      virtual bool equals(const type_t &other) const;

      virtual void printDeclaration(printer &pout) const;
    };
  }
}

#endif
