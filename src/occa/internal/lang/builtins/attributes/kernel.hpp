#ifndef OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_KERNEL_HEADER
#define OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_KERNEL_HEADER

#include <occa/internal/lang/attribute.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      class kernel : public attribute_t {
      public:
        kernel();

        virtual const std::string& name() const;

        virtual bool forFunction() const;
        virtual bool forStatementType(const int sType) const;

        virtual bool isValid(const attributeToken_t &attr) const;
      };
    }
  }
}

#endif
