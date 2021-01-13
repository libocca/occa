#ifndef OCCA_INTERNAL_LANG_STATEMENT_ELIFSTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_ELIFSTATEMENT_HEADER

#include <occa/internal/lang/statement/blockStatement.hpp>

namespace occa {
  namespace lang {
    class elifStatement : public blockStatement {
    public:
      statement_t *condition;

      elifStatement(blockStatement *up_,
                    token_t *source_);
      elifStatement(blockStatement *up_,
                    const elifStatement &other);
      ~elifStatement();

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
