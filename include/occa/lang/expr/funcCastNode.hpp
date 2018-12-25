#ifndef OCCA_LANG_EXPR_FUNCCASTNODE_HEADER
#define OCCA_LANG_EXPR_FUNCCASTNODE_HEADER

#include <occa/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class funcCastNode : public exprNode {
    public:
      vartype_t valueType;
      exprNode *value;

      funcCastNode(token_t *token_,
                   const vartype_t &valueType_,
                   const exprNode &value_);

      funcCastNode(const funcCastNode &node);

      virtual ~funcCastNode();

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
