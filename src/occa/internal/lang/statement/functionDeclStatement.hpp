#ifndef OCCA_INTERNAL_LANG_STATEMENT_FUNCTIONDECLSTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_FUNCTIONDECLSTATEMENT_HEADER

#include <occa/internal/lang/statement/blockStatement.hpp>

namespace occa {
  namespace lang {
    class functionNode;

    class functionDeclStatement : public blockStatement {
    public:
      functionNode *funcNode;

      functionDeclStatement(blockStatement *up_,
                            function_t &function_);

      functionDeclStatement(blockStatement *up_,
                            const functionDeclStatement &other);

      ~functionDeclStatement();

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      function_t& function();
      const function_t& function() const;

      bool addFunctionToParentScope();
      void addArgumentsToScope();

      exprNodeArray getDirectExprNodes();

      virtual void safeReplaceExprNode(exprNode *currentNode, exprNode *newNode);

      virtual void print(printer &pout) const;
    };
  }
}

#endif
