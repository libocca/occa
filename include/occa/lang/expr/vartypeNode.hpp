#ifndef OCCA_INTERNAL_LANG_EXPR_VARTYPENODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_VARTYPENODE_HEADER

#include <occa/internal/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class vartypeNode : public exprNode {
    public:
      vartype_t value;

      vartypeNode(token_t *token_,
                  const vartype_t &value_);

      vartypeNode(const vartypeNode& node);

      virtual ~vartypeNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual bool hasAttribute(const std::string &attr) const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
