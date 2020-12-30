#include <occa/internal/lang/statement/classAccessStatement.hpp>

namespace occa {
  namespace lang {
    classAccessStatement::classAccessStatement(blockStatement *up_,
                                               token_t *source_,
                                               const int access_) :
      statement_t(up_, source_),
      access(access_) {}

    classAccessStatement::classAccessStatement(blockStatement *up_,
                                               const classAccessStatement &other) :
      statement_t(up_, other),
      access(other.access) {}

    classAccessStatement::~classAccessStatement() {}

    statement_t& classAccessStatement::clone_(blockStatement *up_) const {
      return *(new classAccessStatement(up_, *this));
    }

    int classAccessStatement::type() const {
      return statementType::classAccess;
    }

    std::string classAccessStatement::statementName() const {
      return "class access";
    }

    void classAccessStatement::print(printer &pout) const {
      pout.removeIndentation();

      pout.printIndentation();
      if (access & classAccess::public_) {
        pout << "public:\n";
      }
      else if (access & classAccess::private_) {
        pout << "private:\n";
      }
      else if (access & classAccess::protected_) {
        pout << "protected:\n";
      }

      pout.addIndentation();
    }
  }
}
