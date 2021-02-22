#ifndef OCCA_INTERNAL_LANG_EXPR_LAMBDANODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_LAMBDANODE_HEADER

#include <occa/internal/lang/expr/exprNode.hpp>
#include <occa/internal/lang/expr/functionNode.hpp>

namespace occa {
  namespace lang {
    class lambdaNode : public exprNode {
    public:
      lambda_t &value;

      lambdaNode(token_t *token_,
                 lambda_t &value_);

      lambdaNode(const lambdaNode& node);

      virtual ~lambdaNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual bool hasAttribute(const std::string &attr) const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
} 

#endif
