#include <occa/internal/lang/variable.hpp>
#include <occa/internal/lang/expr/dpcppAtomicNode.hpp>
#include <occa/internal/utils/string.hpp>

namespace occa
{
  namespace lang
  {

    dpcppAtomicNode::dpcppAtomicNode(token_t *token_,
                                     const vartype_t& atomic_type_,
                                     const exprNode& value_)
        : exprNode{token_}, atomic_type{atomic_type_}, value{value_.clone()}
    {
    }

    dpcppAtomicNode::dpcppAtomicNode(const dpcppAtomicNode &node)
    : exprNode{node.token}, atomic_type{node.atomic_type}, value{node.value->clone()}
    {}

    dpcppAtomicNode::~dpcppAtomicNode()
    {
      delete value;
    }

    exprNode *dpcppAtomicNode::clone() const
    {
      return new dpcppAtomicNode(*this);
    }

    // `atomic_ref<T,memory_order::relaxed,memory_scope::device,address_space::XXX_space>(expr)`
    void dpcppAtomicNode::print(printer &pout) const
    {

      pout << "sycl::atomic_ref<";

      // Currently CUDA only supports atomics on fundamental types:
      // assume that we can safefuly ignore the pointer types for now
      // and simply print the typename.
      pout << atomic_type.name() << ",";
      pout << "sycl::memory_order::relaxed,";

      //  The SYCL standard states,
      // 
      //  > Using any broader scope for atomic operations in 
      //  > work-group local memory is treated as though 
      //  > `memory_scope::work_group` was specified. 
      // 
      //  Currently OCCA does not address system-wide atomics;
      //  therefore, assume for now that we can always safely
      //  use `memory_scope::device`.
      pout << "sycl::memory_scope::device,";

      if(atomic_type.hasAttribute("shared"))
      {
        pout << "sycl::access::address_space::local_space";
      }
      else
      {
        pout << "sycl::access::address_space::global_space";
      }

      pout << ">(";
      pout << *value;
      pout << ")";
    }

    void dpcppAtomicNode::debugPrint(const std::string &prefix) const
    {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << (*this);
      io::stderr << "] (dpcppAtomic)\n";
    }

  } // namespace lang
} // namespace occa

