#ifndef OCCA_LANG_EXPR_TUPLENODE_HEADER
#define OCCA_LANG_EXPR_TUPLENODE_HEADER

#include <occa/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class tupleNode : public exprNode {
    public:
      exprNodeVector args;

      tupleNode(token_t *token_,
                const exprNodeVector &args_);

      tupleNode(const tupleNode &node);

      virtual ~tupleNode();

      inline int argCount() const {
        return (int) args.size();
      }

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
