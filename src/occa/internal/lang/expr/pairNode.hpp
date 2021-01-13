#ifndef OCCA_INTERNAL_LANG_EXPR_PAIRNODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_PAIRNODE_HEADER

#include <occa/internal/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class pairNode : public exprNode {
    public:
      const operator_t &op;
      exprNode *value;

      pairNode(operatorToken &opToken,
               const exprNode &value_);

      pairNode(const pairNode &node);

      virtual ~pairNode();

      virtual udim_t type() const;
      opType_t opType() const;

      virtual exprNode* clone() const;

      virtual exprNode* startNode();
      virtual exprNode* endNode();

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual void pushChildNodes(exprNodeVector &children);

      virtual bool safeReplaceExprNode(exprNode *currentNode, exprNode *newNode);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
