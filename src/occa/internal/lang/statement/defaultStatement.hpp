#ifndef OCCA_INTERNAL_LANG_STATEMENT_DEFAULTSTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_DEFAULTSTATEMENT_HEADER

#include <occa/internal/lang/statement/statement.hpp>

namespace occa {
  namespace lang {
    class defaultStatement : public statement_t {
    public:
      defaultStatement(blockStatement *up_,
                       token_t *source_);
      defaultStatement(blockStatement *up_,
                       const defaultStatement &other);
      ~defaultStatement();

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      virtual void print(printer &pout) const;
    };
  }
}

#endif
