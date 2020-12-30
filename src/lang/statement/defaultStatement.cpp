#include <occa/internal/lang/statement/defaultStatement.hpp>

namespace occa {
  namespace lang {
    defaultStatement::defaultStatement(blockStatement *up_,
                                       token_t *source_) :
      statement_t(up_, source_) {}

    defaultStatement::defaultStatement(blockStatement *up_,
                                       const defaultStatement &other) :
      statement_t(up_, other) {}

    defaultStatement::~defaultStatement() {}

    statement_t& defaultStatement::clone_(blockStatement *up_) const {
      return *(new defaultStatement(up_, *this));
    }

    int defaultStatement::type() const {
      return statementType::default_;
    }

    std::string defaultStatement::statementName() const {
      return "default";
    }

    void defaultStatement::print(printer &pout) const {
      pout.removeIndentation();

      pout.printIndentation();
      pout << "default:\n";

      pout.addIndentation();
    }
  }
}
