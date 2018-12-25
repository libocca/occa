#ifndef OCCA_LANG_STATEMENT_FUNCTIONSTATEMENT_HEADER
#define OCCA_LANG_STATEMENT_FUNCTIONSTATEMENT_HEADER

#include <occa/lang/statement/statement.hpp>

namespace occa {
  namespace lang {
    class functionStatement : public statement_t {
    public:
      function_t &function;

      functionStatement(blockStatement *up_,
                        function_t &function_);
      functionStatement(blockStatement *up_,
                        const functionStatement&other);
      ~functionStatement();

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      virtual void print(printer &pout) const;
    };
  }
}

#endif
