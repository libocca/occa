#ifndef OCCA_LANG_STATEMENT_STRUCTSTATEMENT_HEADER
#define OCCA_LANG_STATEMENT_STRUCTSTATEMENT_HEADER

#include <occa/lang/statement/statement.hpp>
#include <occa/lang/type/struct.hpp>

namespace occa {
  namespace lang {
    class structStatement : public statement_t {
    public:
      struct_t &struct_;

      structStatement(blockStatement *up_,
                      struct_t &struct_);
      structStatement(blockStatement *up_,
                      const structStatement& other);

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      bool addStructToParentScope();

      virtual void print(printer &pout) const;
    };
  }
}

#endif
