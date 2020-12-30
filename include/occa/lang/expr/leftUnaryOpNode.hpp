#ifndef OCCA_INTERNAL_LANG_EXPR_LEFTUNARYOPNODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_LEFTUNARYOPNODE_HEADER

#include <occa/internal/lang/expr/exprOpNode.hpp>

namespace occa {
  namespace lang {
    class leftUnaryOpNode : public exprOpNode {
    public:
      exprNode *value;

      leftUnaryOpNode(token_t *token_,
                      const unaryOperator_t &op_,
                      const exprNode &value_);

      leftUnaryOpNode(const leftUnaryOpNode &node);

      virtual ~leftUnaryOpNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

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
