#ifndef OCCA_INTERNAL_LANG_EXPR_CUDACALLNODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_CUDACALLNODE_HEADER

#include <occa/internal/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class cudaCallNode : public exprNode {
    public:
      exprNode *value;
      exprNode *blocks, *threads;

      cudaCallNode(token_t *token_,
                   const exprNode &value_,
                   const exprNode &blocks_,
                   const exprNode &threads_);

      cudaCallNode(const cudaCallNode &node);

      virtual ~cudaCallNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* startNode();
      virtual exprNode* endNode();

      virtual void pushChildNodes(exprNodeVector &children);

      virtual bool safeReplaceExprNode(exprNode *currentNode, exprNode *newNode);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
