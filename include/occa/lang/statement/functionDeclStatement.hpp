#ifndef OCCA_LANG_STATEMENT_FUNCTIONDECLSTATEMENT_HEADER
#define OCCA_LANG_STATEMENT_FUNCTIONDECLSTATEMENT_HEADER

#include <occa/lang/statement/blockStatement.hpp>

namespace occa {
  namespace lang {
    class functionDeclStatement : public blockStatement {
    public:
      function_t &function;

      functionDeclStatement(blockStatement *up_,
                            function_t &function_);
      functionDeclStatement(blockStatement *up_,
                            const functionDeclStatement &other);

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      bool addFunctionToParentScope();
      void addArgumentsToScope();

      virtual void print(printer &pout) const;
    };
  }
}

#endif
