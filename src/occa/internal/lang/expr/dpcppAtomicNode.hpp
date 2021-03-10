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

    private:
      inline static const std::string sycl_atomic_ref{"sycl::ONEAPI::atomic_ref"};
      inline static const std::string memory_order_relaxed{"sycl::ONEAPI::memory_order::relaxed"};

      inline static const std::string memory_scope_device{"sycl::ONEAPI::memory_scope::device"};

      inline static const std::string address_space_global_space{"sycl::access::address_space::global_space"};
      inline static const std::string address_space_local_space{"sycl::access::address_space::local_space"};
    };
  }
}

#endif
