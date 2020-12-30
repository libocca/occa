#ifndef OCCA_INTERNAL_LANG_STATEMENT_EMPTYSTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_EMPTYSTATEMENT_HEADER

#include <occa/internal/lang/statement/statement.hpp>

namespace occa {
  namespace lang {
    class emptyStatement : public statement_t {
    public:
      bool hasSemicolon;

      emptyStatement(blockStatement *up_,
                     token_t *source_,
                     const bool hasSemicolon_ = true);
      emptyStatement(blockStatement *up_,
                     const emptyStatement &other);
      ~emptyStatement();

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      virtual void print(printer &pout) const;
    };
  }
}

#endif
