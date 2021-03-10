#ifndef OCCA_INTERNAL_LANG_EXPR_FUNCTIONNODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_FUNCTIONNODE_HEADER

#include <occa/internal/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class functionNode : public exprNode {
    public:
      function_t &value;

      functionNode(token_t *token_,
                   function_t &value_);

      functionNode(const functionNode& node);

      virtual ~functionNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual bool hasAttribute(const std::string &attr) const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
