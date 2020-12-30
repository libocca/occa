#ifndef OCCA_INTERNAL_LANG_STATEMENT_GOTOLABELSTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_GOTOLABELSTATEMENT_HEADER

#include <occa/internal/lang/statement/statement.hpp>

namespace occa {
  namespace lang {
    class gotoLabelStatement : public statement_t {
    public:
      identifierToken &labelToken;

      gotoLabelStatement(blockStatement *up_,
                         identifierToken &labelToken_);
      gotoLabelStatement(blockStatement *up_,
                         const gotoLabelStatement &other);
      ~gotoLabelStatement();

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
