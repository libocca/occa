#include <occa/internal/lang/expr/dpcppLocalMemoryNode.hpp>
#include <occa/internal/utils/string.hpp>

namespace occa
{
  namespace lang
  {

    dpcppLocalMemoryNode::dpcppLocalMemoryNode(token_t *token_,
                                               const vartype_t &shared_type_,
                                               const std::string &handler_name_)
        : exprNode{token_}, shared_type{shared_type_}, handler_name{handler_name_}
    {
    }

    dpcppLocalMemoryNode::dpcppLocalMemoryNode(const dpcppLocalMemoryNode &node)
        : exprNode{node.token}, shared_type{node.shared_type}, handler_name{node.handler_name}
    {
    }

    exprNode *dpcppLocalMemoryNode::clone() const
    {
      return new dpcppLocalMemoryNode(*this);
    }

    void dpcppLocalMemoryNode::print(printer &pout) const
    {
      pout << "*(sycl::ext::oneapi::group_local_memory_for_overwrite<";
      pout << shared_type;
      pout << ">(";
      pout << handler_name;
      pout << ".get_group()))";
    }

    void dpcppLocalMemoryNode::debugPrint(const std::string &prefix) const
    {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                 << prefix << "|---[";
      pout << (*this);
      io::stderr << "] (dpcppLocalMemory)\n";
    }

  } // namespace lang
} // namespace occa

