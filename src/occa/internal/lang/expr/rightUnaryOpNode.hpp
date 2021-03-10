#ifndef OCCA_INTERNAL_LANG_EXPR_RIGHTUNARYOPNODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_RIGHTUNARYOPNODE_HEADER

#include <occa/internal/lang/expr/exprOpNode.hpp>

namespace occa {
  namespace lang {
    class rightUnaryOpNode : public exprOpNode {
    public:
      exprNode *value;

      rightUnaryOpNode(const unaryOperator_t &op_,
                       const exprNode &value_);

      rightUnaryOpNode(token_t *token,
                       const unaryOperator_t &op_,
                       const exprNode &value_);

      rightUnaryOpNode(const rightUnaryOpNode &node);

      virtual ~rightUnaryOpNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual exprNode* startNode();

      virtual void pushChildNodes(exprNodeVector &children);

      virtual bool safeReplaceExprNode(exprNode *currentNode, exprNode *newNode);

      virtual variable_t* getVariable();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
