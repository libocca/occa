#ifndef OCCA_INTERNAL_LANG_EXPR_EMPTYNODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_EMPTYNODE_HEADER

#include <occa/internal/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class emptyNode : public exprNode {
    public:
      emptyNode();
      virtual ~emptyNode();

      virtual udim_t type() const;

      virtual exprNode* clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };

    extern const emptyNode noExprNode;
  }
}

#endif
