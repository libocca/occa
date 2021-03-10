#include <occa/internal/lang/statement/caseStatement.hpp>
#include <occa/internal/lang/expr.hpp>

namespace occa {
  namespace lang {
    caseStatement::caseStatement(blockStatement *up_,
                                 token_t *source_,
                                 exprNode &value_) :
      statement_t(up_, source_),
      value(&value_) {}

    caseStatement::caseStatement(blockStatement *up_,
                                 const caseStatement &other) :
      statement_t(up_, other),
      value(other.value->clone()) {}

    caseStatement::~caseStatement() {
      delete value;
    }

    statement_t& caseStatement::clone_(blockStatement *up_) const {
      return *(new caseStatement(up_, *this));
    }

    int caseStatement::type() const {
      return statementType::case_;
    }

    std::string caseStatement::statementName() const {
      return "case";
    }

    void caseStatement::print(printer &pout) const {
      pout.removeIndentation();

      pout.printIndentation();
      pout << "case ";
      pout.pushInlined(true);
      pout << *value;
      pout.popInlined();
      pout << ":\n";

      pout.addIndentation();
    }
  }
}
