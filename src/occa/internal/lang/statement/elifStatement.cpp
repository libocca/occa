#include <occa/internal/lang/statement/elifStatement.hpp>

namespace occa {
  namespace lang {
    elifStatement::elifStatement(blockStatement *up_,
                                 token_t *source_) :
      blockStatement(up_, source_),
      condition(NULL) {}

    elifStatement::elifStatement(blockStatement *up_,
                                 const elifStatement &other) :
      blockStatement(up_, other.source),
      condition(&(other.condition->clone(this))) {
      copyFrom(other);
    }

    elifStatement::~elifStatement() {
      delete condition;
    }

    void elifStatement::setCondition(statement_t *condition_) {
      condition = condition_;
    }

    statement_t& elifStatement::clone_(blockStatement *up_) const {
      return *(new elifStatement(up_, *this));
    }

    int elifStatement::type() const {
      return statementType::elif_;
    }

    std::string elifStatement::statementName() const {
      return "else if";
    }

    statementArray elifStatement::getInnerStatements() {
      statementArray arr;

      if (condition) {
        arr.push(condition);
      }

      return arr;
    }

    void elifStatement::print(printer &pout) const {
      pout.printStartIndentation();
      pout << "else if (";
      pout.pushInlined(true);
      condition->print(pout);
      pout << ')';

      blockStatement::print(pout);
      pout.popInlined();
    }
  }
}
