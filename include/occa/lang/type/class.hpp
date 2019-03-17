#ifndef OCCA_LANG_TYPE_CLASS_HEADER
#define OCCA_LANG_TYPE_CLASS_HEADER

#include <occa/lang/type/structure.hpp>

namespace occa {
  namespace lang {
    class class_t : public structure_t {
    public:
      class_t();

      virtual int type() const;
      virtual type_t& clone() const;

      virtual dtype_t dtype() const;

      virtual void printDeclaration(printer &pout) const;
    };
  }
}

#endif
