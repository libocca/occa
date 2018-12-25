#ifndef OCCA_LANG_EXPR_IDENTIFIERNODE_HEADER
#define OCCA_LANG_EXPR_IDENTIFIERNODE_HEADER

#include <occa/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class identifierNode : public exprNode {
    public:
      std::string value;

      identifierNode(token_t *token_,
                     const std::string &value_);

      identifierNode(const identifierNode& node);

      virtual ~identifierNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
