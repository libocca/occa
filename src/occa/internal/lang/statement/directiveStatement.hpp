#ifndef OCCA_INTERNAL_LANG_STATEMENT_DIRECTIVESTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_DIRECTIVESTATEMENT_HEADER

#include <occa/internal/lang/statement/statement.hpp>

namespace occa {
  namespace lang {
    class directiveStatement : public statement_t {
    public:
      directiveToken &token;

      directiveStatement(blockStatement *up_,
                         const directiveToken &token_);
      directiveStatement(blockStatement *up_,
                         const directiveStatement &other);
      ~directiveStatement();

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      std::string& value();
      const std::string& value() const;

      virtual void print(printer &pout) const;
    };
  }
}

#endif
