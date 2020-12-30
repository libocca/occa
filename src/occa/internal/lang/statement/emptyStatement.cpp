#include <occa/internal/lang/statement/emptyStatement.hpp>

namespace occa {
  namespace lang {
    emptyStatement::emptyStatement(blockStatement *up_,
                                   token_t *source_,
                                   const bool hasSemicolon_) :
      statement_t(up_, source_),
      hasSemicolon(hasSemicolon_) {}

    emptyStatement::emptyStatement(blockStatement *up_,
                                   const emptyStatement &other):
      statement_t(up_, other),
      hasSemicolon(other.hasSemicolon) {}

    emptyStatement::~emptyStatement() {}

    statement_t& emptyStatement::clone_(blockStatement *up_) const {
      return *(new emptyStatement(up_, *this));
    }

    int emptyStatement::type() const {
      return statementType::empty;
    }

    std::string emptyStatement::statementName() const {
      return "empty";
    }

    void emptyStatement::print(printer &pout) const {
      if (hasSemicolon) {
        pout.printStartIndentation();
        pout << ';';
        pout.printEndNewline();
      }
    }
  }
}
