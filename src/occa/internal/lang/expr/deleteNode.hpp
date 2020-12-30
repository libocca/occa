#ifndef OCCA_INTERNAL_LANG_EXPR_DELETENODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_DELETENODE_HEADER

#include <occa/internal/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class deleteNode : public exprNode {
    public:
      exprNode *value;
      bool isArray;

      deleteNode(token_t *token_,
                 const exprNode &value_,
                 const bool isArray_);

      deleteNode(const deleteNode &node);

      virtual ~deleteNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* endNode();

      virtual void pushChildNodes(exprNodeVector &children);

      virtual bool safeReplaceExprNode(exprNode *currentNode, exprNode *newNode);

      virtual exprNode* wrapInParentheses();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
