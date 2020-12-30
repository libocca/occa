#ifndef OCCA_INTERNAL_LANG_STATEMENT_NAMESPACESTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_NAMESPACESTATEMENT_HEADER

#include <occa/internal/lang/statement/blockStatement.hpp>

namespace occa {
  namespace lang {
    class namespaceStatement : public blockStatement {
    public:
      identifierToken &nameToken;

      namespaceStatement(blockStatement *up_,
                         identifierToken &nameToken_);
      namespaceStatement(blockStatement *up_,
                         const namespaceStatement &other);
      ~namespaceStatement();

      std::string& name();
      const std::string& name() const;

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      virtual void print(printer &pout) const;
    };
  }
}

#endif
