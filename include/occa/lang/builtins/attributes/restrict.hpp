#ifndef OCCA_LANG_BUILTINS_ATTRIBUTES_RESTRICT_HEADER
#define OCCA_LANG_BUILTINS_ATTRIBUTES_RESTRICT_HEADER

#include <occa/lang/attribute.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      class occaRestrict : public attribute_t {
      public:
        occaRestrict();

        virtual const std::string& name() const;

        virtual bool forVariable() const;
        virtual bool forStatement(const int sType) const;

        virtual bool isValid(const attributeToken_t &attr) const;
      };
    }
  }
}

#endif
