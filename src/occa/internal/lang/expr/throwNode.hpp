#ifndef OCCA_INTERNAL_LANG_EXPR_THROWNODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_THROWNODE_HEADER

#include <occa/internal/lang/expr/exprNode.hpp>
#include <occa/internal/lang/expr/parenthesesNode.hpp>

namespace occa {
  namespace lang {
    class throwNode : public exprNode {
    public:
      exprNode *value;

      throwNode(token_t *token_,
                const exprNode &value_);

      throwNode(const throwNode &node);

      virtual ~throwNode();

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
