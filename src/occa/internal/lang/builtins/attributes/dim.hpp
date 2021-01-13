#ifndef OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_DIM_HEADER
#define OCCA_INTERNAL_LANG_BUILTINS_ATTRIBUTES_DIM_HEADER

#include <occa/internal/lang/attribute.hpp>

namespace occa {
  namespace lang {
    class blockStatement;
    class callNode;

    namespace attributes {
      //---[ @dim ]-----------------------
      class dim : public attribute_t {
      public:
        dim();

        virtual const std::string& name() const;

        virtual bool forVariable() const;
        virtual bool forStatementType(const int sType) const;

        virtual bool isValid(const attributeToken_t &attr) const;

        static bool applyCodeTransformations(blockStatement &root);

        static bool getDimOrder(attributeToken_t &dimAttr,
                                attributeToken_t &dimOrderAttr,
                                intVector &order);

        static bool callHasValidIndices(callNode &call, attributeToken_t &dimAttr);
      };
      //==================================

      //---[ @dimOrder ]------------------
      class dimOrder : public attribute_t {
      public:
        dimOrder();

        virtual const std::string& name() const;

        virtual bool forVariable() const;
        virtual bool forStatementType(const int sType) const;

        virtual bool isValid(const attributeToken_t &attr) const;

        std::string inRangeMessage(const int count) const;
      };
      //==================================
    }
  }
}

#endif
