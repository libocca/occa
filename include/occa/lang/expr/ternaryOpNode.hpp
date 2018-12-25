#ifndef OCCA_LANG_EXPR_TERNARYOPNODE_HEADER
#define OCCA_LANG_EXPR_TERNARYOPNODE_HEADER

#include <occa/lang/expr/exprOpNode.hpp>

namespace occa {
  namespace lang {
    class ternaryOpNode : public exprOpNode {
    public:
      exprNode *checkValue, *trueValue, *falseValue;

      ternaryOpNode(const exprNode &checkValue_,
                    const exprNode &trueValue_,
                    const exprNode &falseValue_);

      ternaryOpNode(const ternaryOpNode &node);
      virtual ~ternaryOpNode();

      virtual udim_t type() const;
      opType_t opType() const;

      virtual exprNode* clone() const;

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual exprNode* startNode();
      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
