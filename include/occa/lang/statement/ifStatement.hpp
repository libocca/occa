#ifndef OCCA_INTERNAL_LANG_STATEMENT_IFSTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_IFSTATEMENT_HEADER

#include <occa/internal/lang/statement/blockStatement.hpp>

namespace occa {
  namespace lang {
    class ifStatement : public blockStatement {
    public:
      statement_t *condition;

      elifStatementVector elifSmnts;
      elseStatement *elseSmnt;

      ifStatement(blockStatement *up_,
                  token_t *source_);
      ifStatement(blockStatement *up_,
                  const ifStatement &other);
      ~ifStatement();

      void setCondition(statement_t *condition_);

      void addElif(elifStatement &elifSmnt);
      void addElse(elseStatement &elseSmnt_);

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      virtual statementArray getInnerStatements();

      virtual void print(printer &pout) const;
    };
  }
}

#endif
