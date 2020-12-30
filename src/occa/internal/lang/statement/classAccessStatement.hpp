#ifndef OCCA_INTERNAL_LANG_STATEMENT_CLASSACCESSSTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_CLASSACCESSSTATEMENT_HEADER

#include <occa/internal/lang/statement/statement.hpp>

namespace occa {
  namespace lang {
    class classAccessStatement : public statement_t {
    public:
      int access;

      classAccessStatement(blockStatement *up_,
                           token_t *source_,
                           const int access_);
      classAccessStatement(blockStatement *up_,
                           const classAccessStatement &other);
      ~classAccessStatement();

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      virtual void print(printer &pout) const;
    };
  }
}

#endif
