#ifndef OCCA_INTERNAL_LANG_EXPR_SUBSCRIPTNODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_SUBSCRIPTNODE_HEADER

#include <occa/internal/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class subscriptNode : public exprNode {
    public:
      exprNode *value, *index;

      subscriptNode(token_t *token_,
                    const exprNode &value_,
                    const exprNode &index_);

      subscriptNode(const subscriptNode &node);

      virtual ~subscriptNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

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
