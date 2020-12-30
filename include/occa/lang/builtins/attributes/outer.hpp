#ifndef OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_OUTER_HEADER
#define OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_OUTER_HEADER

#include <occa/internal/lang/attribute.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      class outer : public attribute_t {
      public:
        outer();

        virtual const std::string& name() const;

        virtual bool forStatementType(const int sType) const;

        virtual bool isValid(const attributeToken_t &attr) const;
      };
    }
  }
}

#endif
