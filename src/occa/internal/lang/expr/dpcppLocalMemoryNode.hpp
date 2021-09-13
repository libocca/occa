#ifndef OCCA_INTERNAL_LANG_EXPR_dpcppLocalMemoryNode_HEADER
#define OCCA_INTERNAL_LANG_EXPR_dpcppLocalMemoryNode_HEADER

#include <occa/internal/lang/expr/exprNode.hpp>

namespace occa
{
  namespace lang
  {
    /**
     * Allocates local memory (SLM) using the DPCPP
     * extension `group_local_memory_for_overwrite`.
     * 
     * \note: dpcppParser uses this class on the RHS of 
     * variableDeclarations with type `auto &`.
     */
    class dpcppLocalMemoryNode : public exprNode
    {
    public:
      vartype_t shared_type;
      std::string handler_name;

      dpcppLocalMemoryNode(token_t *token_,
                           const vartype_t &shared_type_,
                           const std::string &handler_name_);

      dpcppLocalMemoryNode(const dpcppLocalMemoryNode &node);

      ~dpcppLocalMemoryNode() = default;

      inline udim_t type() const { return exprNodeType::dpcppLocalMemory; }

      virtual exprNode *clone() const;

      virtual void print(printer &pout) const;

      virtual void debugPrint(const std::string &prefix) const;
    };
  }
}

#endif
