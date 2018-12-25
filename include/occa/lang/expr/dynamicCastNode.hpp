#ifndef OCCA_LANG_EXPR_DYNAMICCASTNODE_HEADER
#define OCCA_LANG_EXPR_DYNAMICCASTNODE_HEADER

#include <occa/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class dynamicCastNode : public exprNode {
    public:
      vartype_t valueType;
      exprNode *value;

      dynamicCastNode(token_t *token_,
                      const vartype_t &valueType_,
                      const exprNode &value_);

      dynamicCastNode(const dynamicCastNode &node);

      virtual ~dynamicCastNode();

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
