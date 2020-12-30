#ifndef OCCA_INTERNAL_LANG_STATEMENT_GOTOSTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_GOTOSTATEMENT_HEADER

#include <occa/internal/lang/statement/statement.hpp>

namespace occa {
  namespace lang {
    class gotoStatement : public statement_t {
    public:
      identifierToken &labelToken;

      gotoStatement(blockStatement *up_,
                    identifierToken &labelToken_);
      gotoStatement(blockStatement *up_,
                    const gotoStatement &other);
      ~gotoStatement();

      std::string& label();
      const std::string& label() const;

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      virtual void print(printer &pout) const;
    };
  }
}

#endif
