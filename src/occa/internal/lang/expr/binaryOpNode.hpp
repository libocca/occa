#ifndef OCCA_INTERNAL_LANG_EXPR_BINARYOPNODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_BINARYOPNODE_HEADER

#include <occa/internal/lang/expr/exprOpNode.hpp>

namespace occa {
  namespace lang {
    class binaryOpNode : public exprOpNode {
    public:
      exprNode *leftValue, *rightValue;

      binaryOpNode(const binaryOperator_t &op_,
                   const exprNode &leftValue_,
                   const exprNode &rightValue_);

      binaryOpNode(token_t *token,
                   const binaryOperator_t &op_,
                   const exprNode &leftValue_,
                   const exprNode &rightValue_);

      binaryOpNode(const binaryOpNode &node);

      virtual ~binaryOpNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual exprNode* startNode();
      virtual exprNode* endNode();

      virtual void pushChildNodes(exprNodeVector &children);

      virtual bool safeReplaceExprNode(exprNode *currentNode, exprNode *newNode);

      virtual variable_t* getVariable();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
