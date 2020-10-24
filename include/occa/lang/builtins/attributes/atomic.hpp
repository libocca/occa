#ifndef OCCA_LANG_BUILTINS_ATTRIBUTES_ATOMIC_HEADER
#define OCCA_LANG_BUILTINS_ATTRIBUTES_ATOMIC_HEADER

#include <occa/lang/attribute.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      class atomic : public attribute_t {
      public:
        atomic();

        virtual const std::string& name() const;

        virtual bool forStatementType(const int sType) const;

        virtual bool isValid(const attributeToken_t &attr) const;
      };
    }
  }
}

#endif
