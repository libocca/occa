#ifndef OCCA_LANG_TRANSFORMS_BUILTINS_FILLEXPRIDENTIFIERS_HEADER
#define OCCA_LANG_TRANSFORMS_BUILTINS_FILLEXPRIDENTIFIERS_HEADER

#include <occa/lang/transforms/exprTransform.hpp>
#include <occa/lang/transforms/builtins/finders.hpp>

namespace occa {
  namespace lang {
    class blockStatement;

    namespace transforms {
      class fillExprIdentifiers_t : public exprTransform {
      public:
        blockStatement *scopeSmnt;

        fillExprIdentifiers_t(blockStatement *scopeSmnt_);

        virtual exprNode* transformExprNode(exprNode &node);
      };
    }
  }
}

#endif
