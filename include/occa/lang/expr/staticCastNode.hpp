#ifndef OCCA_LANG_EXPR_STATICCASTNODE_HEADER
#define OCCA_LANG_EXPR_STATICCASTNODE_HEADER

#include <occa/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class staticCastNode : public exprNode {
    public:
      vartype_t valueType;
      exprNode *value;

      staticCastNode(token_t *token_,
                     const vartype_t &valueType_,
                     const exprNode &value_);

      staticCastNode(const staticCastNode &node);

      virtual ~staticCastNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* endNode();

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
