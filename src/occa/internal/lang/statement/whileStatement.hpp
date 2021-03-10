#ifndef OCCA_INTERNAL_LANG_STATEMENT_WHILESTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_WHILESTATEMENT_HEADER

#include <occa/internal/lang/statement/blockStatement.hpp>

namespace occa {
  namespace lang {
    class whileStatement : public blockStatement {
    public:
      statement_t *condition;
      bool isDoWhile;

      whileStatement(blockStatement *up_,
                     token_t *source_,
                     const bool isDoWhile_ = false);
      whileStatement(blockStatement *up_,
                     const whileStatement &other);
      ~whileStatement();

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
