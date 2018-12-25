#include <occa/lang/statement/functionStatement.hpp>

namespace occa {
  namespace lang {
    functionStatement::functionStatement(blockStatement *up_,
                                         function_t &function_) :
      statement_t(up_, function_.source),
      function(function_) {}

    functionStatement::functionStatement(blockStatement *up_,
                                         const functionStatement &other) :
      statement_t(up_, other),
      function((function_t&) other.function.clone()) {}

    functionStatement::~functionStatement() {
      // TODO: Add to scope with uniqueName() as the key
      function.free();
      delete &function;
    }

    statement_t& functionStatement::clone_(blockStatement *up_) const {
      return *(new functionStatement(up_, *this));
    }

    int functionStatement::type() const {
      return statementType::function;
    }

    std::string functionStatement::statementName() const {
      return "function";
    }

    void functionStatement::print(printer &pout) const {
      // Double newlines to make it look cleaner
      pout << '\n';
      pout.printStartIndentation();
      function.printDeclaration(pout);
      pout << ";\n";
    }
  }
}
