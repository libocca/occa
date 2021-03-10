#include <occa/internal/lang/statement/returnStatement.hpp>
#include <occa/internal/lang/expr.hpp>

namespace occa {
  namespace lang {
    returnStatement::returnStatement(blockStatement *up_,
                                     token_t *source_,
                                     exprNode *value_) :
      statement_t(up_, source_),
      value(value_) {}

    returnStatement::returnStatement(blockStatement *up_,
                                     const returnStatement &other) :
      statement_t(up_, other),
      value(exprNode::clone(other.value)) {}

    returnStatement::~returnStatement() {
      delete value;
    }

    statement_t& returnStatement::clone_(blockStatement *up_) const {
      return *(new returnStatement(up_, *this));
    }

    int returnStatement::type() const {
      return statementType::return_;
    }

    std::string returnStatement::statementName() const {
      return "return";
    }

    void returnStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "return";
      if (value) {
        pout << ' ';
        pout.pushInlined(true);
        pout << *value;
        pout.popInlined();
      }
      pout << ";\n";
    }
  }
}
