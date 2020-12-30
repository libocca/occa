#include <occa/internal/lang/statement/elseStatement.hpp>

namespace occa {
  namespace lang {
    elseStatement::elseStatement(blockStatement *up_,
                                 token_t *source_) :
      blockStatement(up_, source_) {}

    elseStatement::elseStatement(blockStatement *up_,
                                 const elseStatement &other) :
      blockStatement(up_, other) {}

    statement_t& elseStatement::clone_(blockStatement *up_) const {
      return *(new elseStatement(up_, *this));
    }

    int elseStatement::type() const {
      return statementType::else_;
    }

    std::string elseStatement::statementName() const {
      return "else";
    }

    void elseStatement::print(printer &pout) const {
      pout.printStartIndentation();
      pout << "else";
      pout.pushInlined(true);

      blockStatement::print(pout);
      pout.popInlined();
    }
  }
}
