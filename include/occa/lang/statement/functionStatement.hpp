#ifndef OCCA_INTERNAL_LANG_STATEMENT_FUNCTIONSTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_FUNCTIONSTATEMENT_HEADER

#include <occa/internal/lang/statement/statement.hpp>

namespace occa {
  namespace lang {
    class functionNode;

    class functionStatement : public statement_t {
    public:
      functionNode *funcNode;

      functionStatement(blockStatement *up_,
                        function_t &function_);
      functionStatement(blockStatement *up_,
                        const functionStatement&other);
      ~functionStatement();

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      function_t& function();
      const function_t& function() const;

      virtual exprNodeArray getDirectExprNodes();

      virtual void safeReplaceExprNode(exprNode *currentNode, exprNode *newNode);

      virtual void print(printer &pout) const;
    };
  }
}

#endif
