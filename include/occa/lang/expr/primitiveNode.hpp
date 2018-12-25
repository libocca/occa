#ifndef OCCA_LANG_EXPR_PRIMITIVENODE_HEADER
#define OCCA_LANG_EXPR_PRIMITIVENODE_HEADER

#include <occa/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class primitiveNode : public exprNode {
    public:
      primitive value;

      primitiveNode(token_t *token_,
                    primitive value_);

      primitiveNode(const primitiveNode& node);

      virtual ~primitiveNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
