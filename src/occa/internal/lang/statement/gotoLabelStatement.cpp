#include <occa/internal/lang/statement/gotoLabelStatement.hpp>

namespace occa {
  namespace lang {
    gotoLabelStatement::gotoLabelStatement(blockStatement *up_,
                                           identifierToken &labelToken_) :
      statement_t(up_, &labelToken_),
      labelToken((identifierToken&) *source) {}

    gotoLabelStatement::gotoLabelStatement(blockStatement *up_,
                                           const gotoLabelStatement &other) :
      statement_t(up_, other),
      labelToken((identifierToken&) *source) {}

    gotoLabelStatement::~gotoLabelStatement() {}

    statement_t& gotoLabelStatement::clone_(blockStatement *up_) const {
      return *(new gotoLabelStatement(up_, *this));
    }

    std::string& gotoLabelStatement::label() {
      return labelToken.value;
    }

    const std::string& gotoLabelStatement::label() const {
      return labelToken.value;
    }

    int gotoLabelStatement::type() const {
      return statementType::gotoLabel;
    }

    std::string gotoLabelStatement::statementName() const {
      return "goto label";
    }

    void gotoLabelStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << label() << ":\n";
    }
  }
}
