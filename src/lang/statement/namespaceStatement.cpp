#include <occa/internal/lang/statement/namespaceStatement.hpp>

namespace occa {
  namespace lang {
    namespaceStatement::namespaceStatement(blockStatement *up_,
                                           identifierToken &nameToken_) :
      blockStatement(up_, &nameToken_),
      nameToken((identifierToken&) *source) {}

    namespaceStatement::namespaceStatement(blockStatement *up_,
                                           const namespaceStatement &other) :
      blockStatement(up_, other),
      nameToken((identifierToken&) *source) {}

    namespaceStatement::~namespaceStatement() {}

    statement_t& namespaceStatement::clone_(blockStatement *up_) const {
      return *(new namespaceStatement(up_, *this));
    }

    std::string& namespaceStatement::name() {
      return nameToken.value;
    }

    const std::string& namespaceStatement::name() const {
      return nameToken.value;
    }

    int namespaceStatement::type() const {
      return statementType::namespace_;
    }

    std::string namespaceStatement::statementName() const {
      return "namespace";
    }

    void namespaceStatement::print(printer &pout) const {
      pout.printIndentation();
      pout << "namespace " << name();

      blockStatement::print(pout);
    }
  }
}
