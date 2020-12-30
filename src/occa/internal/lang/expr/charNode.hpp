#ifndef OCCA_INTERNAL_LANG_EXPR_CHARNODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_CHARNODE_HEADER

#include <occa/internal/utils.hpp>
#include <occa/internal/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class charNode : public exprNode {
    public:
      std::string value;

      charNode(token_t *token_,
               const std::string &value_);

      charNode(const charNode& node);

      virtual ~charNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
