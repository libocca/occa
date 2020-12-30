#ifndef OCCA_INTERNAL_LANG_STATEMENT_COMMENTSTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_COMMENTSTATEMENT_HEADER

#include <occa/internal/lang/statement/statement.hpp>

namespace occa {
  namespace lang {
    class commentStatement : public statement_t {
     public:
      commentToken &token;

      commentStatement(blockStatement *up_,
                       const commentToken &token_);
      commentStatement(blockStatement *up_,
                       const commentStatement &other);
      ~commentStatement();

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      virtual void print(printer &pout) const;
    };
  }
}

#endif
