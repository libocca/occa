#ifndef OCCA_INTERNAL_LANG_EXPR_DPCPPATOMICNODE_HEADER
#define OCCA_INTERNAL_LANG_EXPR_DPCPPATOMICNODE_HEADER

#include <occa/internal/lang/expr/exprNode.hpp>

namespace occa
{
  namespace lang
  {
    /**
     * Defines a SYCL atomic reference.
     * 
     * \note Atomic references are only supported
     * by DPC++ and SYCL 2020 or newer.
     */
    class dpcppAtomicNode : public exprNode
    {
    public:
      vartype_t atomic_type;
      exprNode *value{};

      dpcppAtomicNode(token_t *token_,
                      const vartype_t & atomic_type_,
                      const exprNode& value_);

      dpcppAtomicNode(const dpcppAtomicNode &node);

      // ~dpcppAtomicNode() = default;
      ~dpcppAtomicNode();

      inline udim_t type() const { return exprNodeType::dpcppAtomic; }

      virtual exprNode *clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
