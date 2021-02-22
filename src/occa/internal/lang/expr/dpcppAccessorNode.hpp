#ifndef OCCA_INTERNAL_LANG_EXPR_DPCPPACCESSORNODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_DPCPPACCESSORNODE_HEADER

#include <occa/internal/lang/expr/exprNode.hpp>

namespace occa {
  namespace lang {
    class dpcppAccessorNode : public exprNode {
    public:
      vartype_t shared_type;
      std::string handler_name;

      dpcppAccessorNode(token_t *token_,
                    const vartype_t& shared_type_,
                    const std::string& handler_name_);

      dpcppAccessorNode(const dpcppAccessorNode& node);

      ~dpcppAccessorNode() = default;

      inline udim_t type() const { return exprNodeType::dpcppAccessor; }

      virtual exprNode* clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
