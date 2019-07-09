#ifndef OCCA_LANG_BUILTINS_ATTRIBUTES_DIM_HEADER
#define OCCA_LANG_BUILTINS_ATTRIBUTES_DIM_HEADER

#include <occa/lang/attribute.hpp>

namespace occa {
  namespace lang {
    namespace attributes {
      //---[ @dim ]-----------------------
      class dim : public attribute_t {
      public:
        dim();

        virtual const std::string& name() const;

        virtual bool forVariable() const;
        virtual bool forStatement(const int sType) const;

        virtual bool isValid(const attributeToken_t &attr) const;
      };
      //==================================

      //---[ @dimOrder ]------------------
      class dimOrder : public attribute_t {
      public:
        dimOrder();

        virtual const std::string& name() const;

        virtual bool forVariable() const;
        virtual bool forStatement(const int sType) const;

        virtual bool isValid(const attributeToken_t &attr) const;

        std::string inRangeMessage(const int count) const;
      };
      //==================================
    }
  }
}

#endif
