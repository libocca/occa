#ifndef OCCA_LANG_EXPR_THROWNODE_HEADER
#define OCCA_LANG_EXPR_THROWNODE_HEADER

#include <occa/lang/expr/exprNode.hpp>
#include <occa/lang/expr/parenthesesNode.hpp>

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

      virtual void setChildren(exprNodeRefVector &children);

      virtual exprNode* wrapInParentheses();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
