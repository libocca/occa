#ifndef OCCA_INTERNAL_LANG_STATEMENT_PRAGMASTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_PRAGMASTATEMENT_HEADER

#include <occa/internal/lang/statement/statement.hpp>

namespace occa {
  namespace lang {
    class pragmaStatement : public statement_t {
    public:
      pragmaToken &token;

      pragmaStatement(blockStatement *up_,
                      const pragmaToken &token_);
      pragmaStatement(blockStatement *up_,
                      const pragmaStatement &other);
      ~pragmaStatement();

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
