#ifndef OCCA_INTERNAL_LANG_STATEMENT_FORSTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_FORSTATEMENT_HEADER

#include <occa/internal/lang/statement/blockStatement.hpp>

namespace occa {
  namespace lang {
    class forStatement : public blockStatement {
    public:
      statement_t *init, *check, *update;

      forStatement(blockStatement *up_,
                   token_t *source_);

      forStatement(blockStatement *up_,
                   const forStatement &other);

      ~forStatement();

      void setLoopStatements(statement_t *init_,
                             statement_t *check_,
                             statement_t *update_);

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      virtual statementArray getInnerStatements();

      virtual void print(printer &pout) const;
    };
  }
}

#endif
