#include <occa/internal/lang/statement/directiveStatement.hpp>

namespace occa {
  namespace lang {
    directiveStatement::directiveStatement(blockStatement *up_,
                                           const directiveToken &token_) :
      statement_t(up_, &token_),
      token((directiveToken&) *source) {}

    directiveStatement::directiveStatement(blockStatement *up_,
                                           const directiveStatement &other) :
      statement_t(up_, other),
      token((directiveToken&) *source) {}

    directiveStatement::~directiveStatement() {}

    statement_t& directiveStatement::clone_(blockStatement *up_) const {
      return *(new directiveStatement(up_, *this));
    }

    int directiveStatement::type() const {
      return statementType::directive;
    }

    std::string directiveStatement::statementName() const {
      return "directive";
    }

    std::string& directiveStatement::value() {
      return token.value;
    }

    const std::string& directiveStatement::value() const {
      return token.value;
    }

    void directiveStatement::print(printer &pout) const {
      pout << '#' << token.value << '\n';
    }
  }
}
