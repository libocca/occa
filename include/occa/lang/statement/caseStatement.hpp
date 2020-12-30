#ifndef OCCA_INTERNAL_LANG_STATEMENT_CASESTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_CASESTATEMENT_HEADER

#include <occa/internal/lang/statement/statement.hpp>

namespace occa {
  namespace lang {
    class caseStatement : public statement_t {
    public:
      exprNode *value;

      caseStatement(blockStatement *up_,
                    token_t *source_,
                    exprNode &value_);
      caseStatement(blockStatement *up_,
                    const caseStatement &other);
      ~caseStatement();

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      virtual void print(printer &pout) const;
    };
  }
}

#endif
