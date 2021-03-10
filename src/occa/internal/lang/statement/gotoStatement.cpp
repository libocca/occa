#include <occa/internal/lang/statement/gotoStatement.hpp>

namespace occa {
  namespace lang {
    gotoStatement::gotoStatement(blockStatement *up_,
                                 identifierToken &labelToken_) :
      statement_t(up_, &labelToken_),
      labelToken((identifierToken&) *source) {}

    gotoStatement::gotoStatement(blockStatement *up_,
                                 const gotoStatement &other) :
      statement_t(up_, other),
      labelToken((identifierToken&) *source) {}

    gotoStatement::~gotoStatement() {}

    statement_t& gotoStatement::clone_(blockStatement *up_) const {
      return *(new gotoStatement(up_, *this));
    }

    std::string& gotoStatement::label() {
      return labelToken.value;
    }

    const std::string& gotoStatement::label() const {
      return labelToken.value;
    }

    int gotoStatement::type() const {
      return statementType::goto_;
    }

    std::string gotoStatement::statementName() const {
      return "goto";
    }

    void gotoStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "goto " << label() << ';';
    }
  }
}
