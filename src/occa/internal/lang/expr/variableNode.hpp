#ifndef OCCA_INTERNAL_LANG_EXPR_VARIABLENODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_VARIABLENODE_HEADER

#include <occa/internal/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class variableNode : public exprNode {
    public:
      variable_t &value;

      variableNode(token_t *token_,
                   variable_t &value_);

      variableNode(const variableNode& node);

      virtual ~variableNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual bool hasAttribute(const std::string &attr) const;

      virtual variable_t* getVariable();

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
