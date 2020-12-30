#ifndef OCCA_INTERNAL_LANG_STATEMENT_RETURNSTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_RETURNSTATEMENT_HEADER

#include <occa/internal/lang/statement/statement.hpp>

namespace occa {
  namespace lang {
    class returnStatement : public statement_t {
    public:
      exprNode *value;

      returnStatement(blockStatement *up_,
                      token_t *source_,
                      exprNode *value_);
      returnStatement(blockStatement *up_,
                      const returnStatement &other);
      ~returnStatement();

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      virtual void print(printer &pout) const;
    };
  }
}

#endif
