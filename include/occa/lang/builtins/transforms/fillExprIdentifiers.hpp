#ifndef OCCA_LANG_BUILTINS_TRANSFORMS_FILLEXPRIDENTIFIERS_HEADER
#define OCCA_LANG_BUILTINS_TRANSFORMS_FILLEXPRIDENTIFIERS_HEADER

#include <occa/lang/exprTransform.hpp>
#include <occa/lang/builtins/transforms/finders.hpp>

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
