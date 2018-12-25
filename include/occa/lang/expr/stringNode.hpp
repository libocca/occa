#ifndef OCCA_LANG_EXPR_STRINGNODE_HEADER
#define OCCA_LANG_EXPR_STRINGNODE_HEADER

#include <occa/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class stringNode : public exprNode {
    public:
      int encoding;
      std::string value;

      stringNode(token_t *token_,
                 const std::string &value_);

      stringNode(const stringNode& node);

      virtual ~stringNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual void setChildren(exprNodeRefVector &children);

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
