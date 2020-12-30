#ifndef OCCA_INTERNAL_LANG_STATEMENT_ELSESTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_ELSESTATEMENT_HEADER

#include <occa/internal/lang/statement/blockStatement.hpp>

namespace occa {
  namespace lang {
    class elseStatement : public blockStatement {
    public:
      elseStatement(blockStatement *up_,
                    token_t *source_);
      elseStatement(blockStatement *up_,
                    const elseStatement &other);

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      virtual void print(printer &pout) const;
    };
  }
}

#endif
