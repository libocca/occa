#include <occa/internal/lang/expr/dpcppAccessorNode.hpp>
#include <occa/internal/utils/string.hpp>

namespace occa
{
  namespace lang
  {

    dpcppAccessorNode::dpcppAccessorNode(token_t *token_,
                                         const vartype_t &shared_type_,
                                         const std::string &handler_name_)
        : exprNode{token_}, shared_type{shared_type_}, handler_name{handler_name_}
    {
    }

    dpcppAccessorNode::dpcppAccessorNode(const dpcppAccessorNode &node)
    : exprNode{node.token}, shared_type{node.shared_type}, handler_name{node.handler_name}
    {}

    exprNode *dpcppAccessorNode::clone() const
    {
      return new dpcppAccessorNode(*this);
    }

    //@todo: Determine the performance implications of index-ordering
    // E.g., the ordering of the SYCL work-item ID triple is reversed
    // from the OKL `@inner` indices.
    void dpcppAccessorNode::print(printer &pout) const
    {
      const arrayVector &var_dims = shared_type.arrays;
      const std::size_t var_rank = var_dims.size();

      pout << "sycl::accessor<";
      pout << shared_type.name();
      pout << ",";
      pout << occa::toString(var_rank);
      pout << ",";
      pout << "sycl::access::mode::read_write,";
      pout << "sycl::access::target::local>(";

      if (var_rank > 0)
      {
        pout << "{" << *(var_dims[0].size);
        
        if(var_rank > 1) {
          pout << "," << *(var_dims[1].size);

          if(var_rank > 2)
            pout << "," << *(var_dims[2].size);
        }

        pout<< "},";
      }

      pout << handler_name;
      pout << ")";
    }

    void dpcppAccessorNode::debugPrint(const std::string &prefix) const
    {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << (*this);
      io::stderr << "] (dpcppAccessor)\n";
    }

  } // namespace lang
} // namespace occa