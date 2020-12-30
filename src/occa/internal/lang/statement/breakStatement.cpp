#include <occa/internal/lang/statement/breakStatement.hpp>

namespace occa {
  namespace lang {
    breakStatement::breakStatement(blockStatement *up_,
                                   token_t *source_) :
      statement_t(up_, source_) {}

    breakStatement::breakStatement(blockStatement *up_,
                                   const breakStatement &other) :
      statement_t(up_, other) {}

    breakStatement::~breakStatement() {}

    statement_t& breakStatement::clone_(blockStatement *up_) const {
      return *(new breakStatement(up_, *this));
    }

    int breakStatement::type() const {
      return statementType::break_;
    }

    std::string breakStatement::statementName() const {
      return "break";
    }

    void breakStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "break;\n";
    }
  }
}
