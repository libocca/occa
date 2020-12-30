#ifndef OCCA_INTERNAL_LANG_TYPE_PRIMITIVE_HEADER
#define OCCA_INTERNAL_LANG_TYPE_PRIMITIVE_HEADER

#include <occa/internal/lang/type/type.hpp>

namespace occa {
  namespace lang {
    class primitive_t : public type_t {
    public:
      const std::string pname;
      mutable const dtype_t *dtype_;

      primitive_t(const std::string &name_);

      virtual const std::string& name() const;
      virtual bool isNamed() const;

      virtual int type() const;
      virtual type_t& clone() const;

      virtual dtype_t dtype() const;

      virtual void printDeclaration(printer &pout) const;
    };
  }
}

#endif
