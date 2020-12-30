#include <occa/internal/lang/statement/continueStatement.hpp>

namespace occa {
  namespace lang {
    continueStatement::continueStatement(blockStatement *up_,
                                         token_t *source_) :
      statement_t(up_, source_) {}

    continueStatement::continueStatement(blockStatement *up_,
                                         const continueStatement &other) :
      statement_t(up_, other) {}

    continueStatement::~continueStatement() {}

    statement_t& continueStatement::clone_(blockStatement *up_) const {
      return *(new continueStatement(up_, *this));
    }

    int continueStatement::type() const {
      return statementType::continue_;
    }

    std::string continueStatement::statementName() const {
      return "continue";
    }

    void continueStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "continue;\n";
    }
  }
}
