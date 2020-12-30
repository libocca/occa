#ifndef OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_GLOBALPTR_HEADER
#define OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_GLOBALPTR_HEADER

#include <occa/internal/lang/attribute.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      //---[ @globalPtr ]---------------
      class globalPtr : public attribute_t {
      public:
        globalPtr();

        virtual const std::string& name() const;

        virtual bool forVariable() const;
        virtual bool forStatementType(const int sType) const;

        virtual bool isValid(const attributeToken_t &attr) const;
      };
      //================================
    }
  }
}

#endif
