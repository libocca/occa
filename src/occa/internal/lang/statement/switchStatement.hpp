#ifndef OCCA_INTERNAL_LANG_STATEMENT_SWITCHSTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_SWITCHSTATEMENT_HEADER

#include <occa/internal/lang/statement/blockStatement.hpp>

namespace occa {
  namespace lang {
    class switchStatement : public blockStatement {
    public:
      statement_t *condition;

      switchStatement(blockStatement *up_,
                      token_t *source_);
      switchStatement(blockStatement *up_,
                      const switchStatement& other);
      ~switchStatement();

      void setCondition(statement_t *condition_);

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      virtual statementArray getInnerStatements();

      virtual void print(printer &pout) const;
    };
  }
}

#endif
