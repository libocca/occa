#include <occa/lang/statement/blockStatement.hpp>
#include <occa/lang/statement/structStatement.hpp>
#include <occa/lang/variable.hpp>

namespace occa {
  namespace lang {
    structStatement::structStatement(blockStatement *up_,
                                     struct_t &struct__) :
      statement_t(up_, struct__.source),
      struct_(struct__) {}

    structStatement::structStatement(blockStatement *up_,
                                     const structStatement& other) :
      statement_t(up_, other.source),
      struct_((struct_t&) other.struct_.clone()) {}

    statement_t& structStatement::clone_(blockStatement *up_) const {
      return *(new structStatement(up_, *this));
    }

    int structStatement::type() const {
      return statementType::struct_;
    }

    std::string structStatement::statementName() const {
      return "struct";
    }

    bool structStatement::addStructToParentScope() {
      if (up && !up->addToScope(struct_)) {
        return false;
      }
      return true;
    }

    void structStatement::print(printer &pout) const {
      // Double newlines to make it look cleaner
      pout << '\n';
      struct_.printDeclaration(pout);
      pout << '\n';
    }
  }
}
