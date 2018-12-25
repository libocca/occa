#ifndef OCCA_LANG_EXPR_SIZEOFNODE_HEADER
#define OCCA_LANG_EXPR_SIZEOFNODE_HEADER

#include <occa/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class sizeofNode : public exprNode {
    public:
      exprNode *value;

      sizeofNode(token_t *token_,
                 const exprNode &value_);

      sizeofNode(const sizeofNode &node);

      virtual ~sizeofNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* endNode();

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
