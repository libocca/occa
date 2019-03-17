#ifndef OCCA_LANG_TYPE_STRUCT_HEADER
#define OCCA_LANG_TYPE_STRUCT_HEADER

#include <occa/lang/type/structure.hpp>

namespace occa {
  namespace lang {
    class struct_t : public structure_t {
    public:
      struct_t();

      virtual int type() const;
      virtual type_t& clone() const;

      virtual dtype_t dtype() const;

      virtual void printDeclaration(printer &pout) const;
    };
  }
}

#endif
