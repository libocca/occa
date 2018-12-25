#ifndef OCCA_LANG_EXPR_CALLNODE_HEADER
#define OCCA_LANG_EXPR_CALLNODE_HEADER

#include <occa/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class callNode : public exprNode {
    public:
      exprNode *value;
      exprNodeVector args;

      callNode(token_t *token_,
               const exprNode &value_,
               const exprNodeVector &args_);

      callNode(const callNode &node);

      virtual ~callNode();

      inline int argCount() const {
        return (int) args.size();
      }

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
