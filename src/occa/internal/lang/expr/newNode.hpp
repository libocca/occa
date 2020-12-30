#ifndef OCCA_INTERNAL_LANG_EXPR_NEWNODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_NEWNODE_HEADER

#include <occa/internal/lang/expr/exprNode.hpp>
#include <occa/internal/lang/expr/emptyNode.hpp>
#include <occa/internal/lang/expr/parenthesesNode.hpp>

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

      virtual void pushChildNodes(exprNodeVector &children);

      virtual bool safeReplaceExprNode(exprNode *currentNode, exprNode *newNode_);

      virtual exprNode* wrapInParentheses();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
