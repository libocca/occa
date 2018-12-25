#ifndef OCCA_LANG_EXPR_CUDACALLNODE_HEADER
#define OCCA_LANG_EXPR_CUDACALLNODE_HEADER

#include <occa/lang/expr/exprNode.hpp>

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

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
