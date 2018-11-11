#ifndef OCCA_LANG_BUILTINS_ATTRIBUTES_EXCLUSIVE_HEADER
#define OCCA_LANG_BUILTINS_ATTRIBUTES_EXCLUSIVE_HEADER

#include <occa/lang/attribute.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      class exclusive : public attribute_t {
      public:
        exclusive();

        virtual std::string name() const;

        virtual bool forVariable() const;
        virtual bool forStatement(const int sType) const;

        virtual bool isValid(const attributeToken_t &attr) const;
      };
    }
  }
}

#endif
