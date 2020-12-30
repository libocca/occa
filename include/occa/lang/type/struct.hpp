#ifndef OCCA_INTERNAL_LANG_TYPE_STRUCT_HEADER
#define OCCA_INTERNAL_LANG_TYPE_STRUCT_HEADER

#include <occa/internal/lang/type/type.hpp>

namespace occa {
  namespace lang {
    class struct_t : public type_t {
    public:
      variableVector fields;

      struct_t();
      struct_t(identifierToken &nameToken);

      struct_t(const struct_t &other);

      virtual int type() const;
      virtual type_t& clone() const;

      virtual dtype_t dtype() const;

      void addField(variable_t &var);
      void addFields(variableVector &fields_);

      virtual void printDeclaration(printer &pout) const;
    };
  }
}

#endif
