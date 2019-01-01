#ifndef OCCA_LANG_TRANSFORMS_EXPRTRANSFORM_HEADER
#define OCCA_LANG_TRANSFORMS_EXPRTRANSFORM_HEADER

#include <occa/types.hpp>

namespace occa {
  namespace lang {
    class exprNode;

    class exprTransform {
    public:
      udim_t validExprNodeTypes;

      exprTransform();

      virtual exprNode* transformExprNode(exprNode &node) = 0;

      exprNode* apply(exprNode &node);
    };
  }
}

#endif
