#ifndef OCCA_LANG_EXPR_TYPENODE_HEADER
#define OCCA_LANG_EXPR_TYPENODE_HEADER

#include <occa/lang/expr/exprNode.hpp>
#include <occa/lang/type.hpp>

namespace occa {
  namespace lang {
    class typeNode : public exprNode {
    public:
      type_t &value;

      typeNode(token_t *token_,
               type_t &value_);

      typeNode(const typeNode& node);

      virtual ~typeNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual void setChildren(exprNodeRefVector &children);

      virtual bool hasAttribute(const std::string &attr) const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
