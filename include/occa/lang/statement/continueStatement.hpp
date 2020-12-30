#ifndef OCCA_INTERNAL_LANG_STATEMENT_CONTINUESTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_CONTINUESTATEMENT_HEADER

#include <occa/internal/lang/statement/statement.hpp>

namespace occa {
  namespace lang {
    class continueStatement : public statement_t {
    public:
      continueStatement(blockStatement *up_,
                        token_t *source_);
      continueStatement(blockStatement *up_,
                        const continueStatement &other);
      ~continueStatement();

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      virtual void print(printer &pout) const;
    };
  }
}

#endif
