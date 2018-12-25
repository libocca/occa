#include <occa/lang/statement/expressionStatement.hpp>
#include <occa/lang/expr.hpp>

namespace occa {
  namespace lang {
    expressionStatement::expressionStatement(blockStatement *up_,
                                             exprNode &expr_,
                                             const bool hasSemicolon_) :
      statement_t(up_, expr_.startNode()->token),
      expr(&expr_),
      hasSemicolon(hasSemicolon_) {}

    expressionStatement::expressionStatement(blockStatement *up_,
                                             const expressionStatement &other) :
      statement_t(up_, other),
      expr(other.expr->clone()),
      hasSemicolon(other.hasSemicolon) {}

    expressionStatement::~expressionStatement() {
      delete expr;
    }

    statement_t& expressionStatement::clone_(blockStatement *up_) const {
      return *(new expressionStatement(up_, *this));
    }

    int expressionStatement::type() const {
      return statementType::expression;
    }

    std::string expressionStatement::statementName() const {
      return "expression";
    }

    void expressionStatement::print(printer &pout) const {
      pout.printStartIndentation();
      pout << (*expr);
      if (hasSemicolon) {
        pout << ';';
        pout.printEndNewline();
      }
    }
  }
}
