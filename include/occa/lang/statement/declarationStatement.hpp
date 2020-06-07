#ifndef OCCA_LANG_STATEMENT_DECLARATIONSTATEMENT_HEADER
#define OCCA_LANG_STATEMENT_DECLARATIONSTATEMENT_HEADER

#include <occa/lang/statement/statement.hpp>

namespace occa {
  namespace lang {
    class declarationStatement : public statement_t {
    public:
      variableDeclarationVector declarations;
      bool declaredType;

      declarationStatement(blockStatement *up_,
                           token_t *source_);
      declarationStatement(blockStatement *up_,
                           const declarationStatement &other);
      ~declarationStatement();

      void clearDeclarations();
      void freeDeclarations();

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      bool addDeclaration(const variableDeclaration &decl,
                          const bool force = false);

      virtual void print(printer &pout) const;
    };
  }
}

#endif
