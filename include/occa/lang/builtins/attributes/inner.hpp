#ifndef OCCA_LANG_BUILTINS_ATTRIBUTES_INNER_HEADER
#define OCCA_LANG_BUILTINS_ATTRIBUTES_INNER_HEADER

#include <occa/lang/attribute.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      class inner : public attribute_t {
      public:
        inner();

        virtual std::string name() const;

        virtual bool forStatement(const int sType) const;

        virtual bool isValid(const attributeToken_t &attr) const;
      };
    }
  }
}

#endif
