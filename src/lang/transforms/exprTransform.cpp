#include <occa/lang/expr.hpp>
#include <occa/lang/transforms/exprTransform.hpp>

namespace occa {
  namespace lang {
    exprTransform::exprTransform() :
      validExprNodeTypes(0) {}

    exprNode* exprTransform::apply(exprNode &node) {
      // Apply transform to children
      exprNodeRefVector children;
      node.setChildren(children);
      const int childCount = (int) children.size();
      for (int i = 0; i < childCount; ++i) {
        exprNode *&child = *(children[i]);
        exprNode *newChild = apply(*child);
        if (!newChild) {
          return NULL;
        }
        child = newChild;
      }

      // Apply transform to self
      exprNode *newNode = &node;
      if (node.type() & validExprNodeTypes) {
        newNode = transformExprNode(node);
        if (newNode != &node) {
          delete &node;
        }
      }
      return newNode;
    }
  }
}
