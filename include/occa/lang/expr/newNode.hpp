#ifndef OCCA_LANG_EXPR_NEWNODE_HEADER
#define OCCA_LANG_EXPR_NEWNODE_HEADER

#include <occa/lang/expr/exprNode.hpp>
#include <occa/lang/expr/emptyNode.hpp>
#include <occa/lang/expr/parenthesesNode.hpp>

namespace occa {
  namespace lang {
    class newNode : public exprNode {
    public:
      vartype_t valueType;
      exprNode *value;
      exprNode *size;

      newNode(token_t *token_,
              const vartype_t &valueType_,
              const exprNode &value_);

      newNode(token_t *token_,
              const vartype_t &valueType_,
              const exprNode &value_,
              const exprNode &size_);

      newNode(const newNode &node);

      virtual ~newNode();

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
