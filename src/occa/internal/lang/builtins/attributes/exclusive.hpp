#ifndef OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_EXCLUSIVE_HEADER
#define OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_EXCLUSIVE_HEADER

#include <occa/internal/lang/attribute.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      class exclusive : public attribute_t {
      public:
        exclusive();

        virtual const std::string& name() const;

        virtual bool forVariable() const;
        virtual bool forStatementType(const int sType) const;

        virtual bool isValid(const attributeToken_t &attr) const;
      };
    }
  }
}

#endif
