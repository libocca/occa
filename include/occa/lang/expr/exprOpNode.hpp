#ifndef OCCA_INTERNAL_LANG_EXPR_EXPROPNODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_EXPROPNODE_HEADER

#include <occa/internal/lang/expr/exprNode.hpp>
#include <occa/internal/lang/operator.hpp>

namespace occa {
  namespace lang {
    class exprOpNode : public exprNode {
    public:
      const operator_t &op;

      exprOpNode(operatorToken &token_);

      exprOpNode(token_t *token_,
                 const operator_t &op_);

      opType_t opType() const;

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual exprNode* wrapInParentheses();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
