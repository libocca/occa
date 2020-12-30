#include <occa/internal/lang/statement/whileStatement.hpp>

namespace occa {
  namespace lang {
    whileStatement::whileStatement(blockStatement *up_,
                                   token_t *source_,
                                   const bool isDoWhile_) :
      blockStatement(up_, source_),
      condition(NULL),
      isDoWhile(isDoWhile_) {}

    whileStatement::whileStatement(blockStatement *up_,
                                   const whileStatement &other) :
      blockStatement(up_, other.source),
      condition(statement_t::clone(up_, other.condition)),
      isDoWhile(other.isDoWhile) {
      copyFrom(other);
    }

    whileStatement::~whileStatement() {
      delete condition;
    }

    void whileStatement::setCondition(statement_t *condition_) {
      condition = condition_;
    }

    statement_t& whileStatement::clone_(blockStatement *up_) const {
      return *(new whileStatement(up_, *this));
    }

    int whileStatement::type() const {
      return statementType::while_;
    }

    std::string whileStatement::statementName() const {
      return isDoWhile ? "do while" : "while";
    }

    statementArray whileStatement::getInnerStatements() {
      statementArray arr;

      if (condition) {
        arr.push(condition);
      }

      return arr;
    }

    void whileStatement::print(printer &pout) const {
      pout.printStartIndentation();
      if (!isDoWhile) {
        pout << "while (";
        pout.pushInlined(true);
        condition->print(pout);
        pout << ')';
      } else {
        pout << "do";
      }

      blockStatement::print(pout);

      if (isDoWhile) {
        pout.popInlined();
        pout << " while (";
        pout.pushInlined(true);
        condition->print(pout);
        pout.popInlined();
        pout << ");";
      } else {
        pout.popInlined();
      }
      pout.printEndNewline();
    }
  }
}
