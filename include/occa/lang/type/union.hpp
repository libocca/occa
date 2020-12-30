#ifndef OCCA_INTERNAL_LANG_TYPE_UNION_HEADER
#define OCCA_INTERNAL_LANG_TYPE_UNION_HEADER

#include <occa/internal/lang/type/structure.hpp>

namespace occa {
  namespace lang {
    class union_t : public structure_t {
    public:
      union_t();

      virtual int type() const;
      virtual type_t& clone() const;

      virtual dtype_t dtype() const;

      virtual void printDeclaration(printer &pout) const;
    };
  }
}

#endif
