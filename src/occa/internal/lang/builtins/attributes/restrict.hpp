#ifndef OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_RESTRICT_HEADER
#define OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_RESTRICT_HEADER

#include <occa/internal/lang/attribute.hpp>

namespace occa {
  namespace lang {
    class blockStatement;
    class qualifier_t;

    namespace attributes {
      class occaRestrict : public attribute_t {
      public:
        occaRestrict();

        virtual const std::string& name() const;

        virtual bool forVariable() const;
        virtual bool forStatementType(const int sType) const;

        virtual bool isValid(const attributeToken_t &attr) const;

        static bool applyCodeTransformations(blockStatement &root,
                                             const qualifier_t &restrictQualifier);
      };
    }
  }
}

#endif
