#ifndef OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_NOBARRIER_HEADER
#define OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_NOBARRIER_HEADER

#include <occa/internal/lang/attribute.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      //---[ @nobarrier ]---------------
      class noBarrier : public attribute_t {
      public:
        noBarrier();

        virtual const std::string& name() const;

        virtual bool forStatementType(const int sType) const;

        virtual bool isValid(const attributeToken_t &attr) const;
      };
      //================================
    }
  }
}

#endif
