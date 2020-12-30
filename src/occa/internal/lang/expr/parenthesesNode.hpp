#ifndef OCCA_INTERNAL_LANG_EXPR_PARENTHESESNODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_PARENTHESESNODE_HEADER

#include <occa/internal/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class parenthesesNode : public exprNode {
    public:
      exprNode *value;

      parenthesesNode(token_t *token_,
                      const exprNode &value_);

      parenthesesNode(const parenthesesNode &node);

      virtual ~parenthesesNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* startNode();
      virtual exprNode* endNode();

      virtual bool canEvaluate() const;
      virtual primitive evaluate() const;

      virtual void pushChildNodes(exprNodeVector &children);

      virtual bool safeReplaceExprNode(exprNode *currentNode, exprNode *newNode);

      virtual variable_t* getVariable();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
