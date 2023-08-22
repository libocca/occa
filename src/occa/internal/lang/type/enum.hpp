#ifndef OCCA_INTERNAL_LANG_TYPE_ENUM_HEADER
#define OCCA_INTERNAL_LANG_TYPE_ENUM_HEADER

#include <occa/internal/lang/type/type.hpp>

namespace occa {
  namespace lang {
    class enum_t;

    class enum_t : public type_t {
    public:
      enumeratorVector enumerators;

      enum_t();
      enum_t(identifierToken &nameToken);
      enum_t(const enum_t &other);


      virtual int type() const;
      virtual type_t& clone() const;
      virtual dtype_t dtype() const;

      void addEnumerator(enumerator_t &enumerator_);
      void addEnumerators(enumeratorVector &enumerators_);

      void debugPrint() const;

      virtual void printDeclaration(printer &pout) const;
    };
  }
}

#endif
