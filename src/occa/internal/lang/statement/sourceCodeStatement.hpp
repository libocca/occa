#ifndef OCCA_INTERNAL_LANG_STATEMENT_SOURCECODESTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_SOURCECODESTATEMENT_HEADER

#include <occa/internal/lang/statement/statement.hpp>

namespace occa {
  namespace lang {
    class sourceCodeStatement : public statement_t {
     public:
      std::string sourceCode;

      sourceCodeStatement(blockStatement *up_,
                         token_t *sourceToken,
                         const std::string &sourceCode_);
      sourceCodeStatement(blockStatement *up_,
                         const sourceCodeStatement &other);
      ~sourceCodeStatement();

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      virtual void print(printer &pout) const;
    };
  }
}

#endif
