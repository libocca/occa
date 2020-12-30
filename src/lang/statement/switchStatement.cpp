#include <occa/internal/lang/statement/switchStatement.hpp>

namespace occa {
  namespace lang {
    switchStatement::switchStatement(blockStatement *up_,
                                     token_t *source_) :
      blockStatement(up_, source_),
      condition(NULL) {}

    switchStatement::switchStatement(blockStatement *up_,
                                     const switchStatement& other) :
      blockStatement(up_, other.source),
      condition(&(other.condition->clone(this))) {
      copyFrom(other);
    }

    switchStatement::~switchStatement() {
      delete condition;
    }

    void switchStatement::setCondition(statement_t *condition_) {
      condition = condition_;
    }

    statement_t& switchStatement::clone_(blockStatement *up_) const {
      return *(new switchStatement(up_, *this));
    }

    int switchStatement::type() const {
      return statementType::switch_;
    }

    std::string switchStatement::statementName() const {
      return "switch";
    }

    statementArray switchStatement::getInnerStatements() {
      statementArray arr;

      if (condition) {
        arr.push(condition);
      }

      return arr;
    }

    void switchStatement::print(printer &pout) const {
      pout.printStartIndentation();
      pout << "switch (";
      pout.pushInlined(true);
      condition->print(pout);
      pout << ')';

      blockStatement::print(pout);
      pout.popInlined();
    }
  }
}
