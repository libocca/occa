#ifndef OCCA_INTERNAL_LANG_STATEMENT_BREAKSTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_BREAKSTATEMENT_HEADER

#include <occa/internal/lang/statement/statement.hpp>

namespace occa {
  namespace lang {
    class breakStatement : public statement_t {
    public:
      breakStatement(blockStatement *up_,
                     token_t *source_);
      breakStatement(blockStatement *up_,
                     const breakStatement &other);
      ~breakStatement();

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      virtual void print(printer &pout) const;
    };
  }
}

#endif
