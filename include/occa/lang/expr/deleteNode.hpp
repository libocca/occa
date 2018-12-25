#ifndef OCCA_LANG_EXPR_DELETENODE_HEADER
#define OCCA_LANG_EXPR_DELETENODE_HEADER

#include <occa/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class deleteNode : public exprNode {
    public:
      exprNode *value;
      bool isArray;

      deleteNode(token_t *token_,
                 const exprNode &value_,
                 const bool isArray_);

      deleteNode(const deleteNode &node);

      virtual ~deleteNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual exprNode* wrapInParentheses();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
