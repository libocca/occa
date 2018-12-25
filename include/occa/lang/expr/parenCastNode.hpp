#ifndef OCCA_LANG_EXPR_PARENCASTNODE_HEADER
#define OCCA_LANG_EXPR_PARENCASTNODE_HEADER

#include <occa/lang/expr/exprNode.hpp>
#include <occa/lang/expr/parenthesesNode.hpp>

namespace occa {
  namespace lang {
    class parenCastNode : public exprNode {
    public:
      vartype_t valueType;
      exprNode *value;

      parenCastNode(token_t *token_,
                    const vartype_t &valueType_,
                    const exprNode &value_);

      parenCastNode(const parenCastNode &node);

      virtual ~parenCastNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* startNode();
      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual exprNode* wrapInParentheses();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
