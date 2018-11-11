#ifndef OCCA_LANG_BUILTINS_TRANSFORMS_DIM_HEADER
#define OCCA_LANG_BUILTINS_TRANSFORMS_DIM_HEADER

#include <occa/lang/exprTransform.hpp>
#include <occa/lang/statementTransform.hpp>

namespace occa {
  namespace lang {
    namespace transforms {
      class dim : public statementTransform,
                  public exprTransform {
      public:
        blockStatement *scopeSmnt;

        dim();

        virtual statement_t* transformStatement(statement_t &smnt);
        virtual exprNode* transformExprNode(exprNode &node);

        bool isValidDim(callNode &call,
                        attributeToken_t &dimAttr);

        bool getDimOrder(attributeToken_t &dimAttr,
                         attributeToken_t &dimOrderAttr,
                         intVector &order);

        bool applyToDeclStatement(declarationStatement &smnt);
        bool applyToExpr(statement_t &smnt,
                         exprNode *&expr);
      };

      bool applyDimTransforms(statement_t &smnt);
    }
  }
}

#endif
