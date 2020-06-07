#include <occa/lang/statement/functionDeclStatement.hpp>

namespace occa {
  namespace lang {
    functionDeclStatement::functionDeclStatement(blockStatement *up_,
                                                 function_t &function_) :
      blockStatement(up_, function_.source),
      function(function_) {
      addArgumentsToScope();
    }

    functionDeclStatement::functionDeclStatement(blockStatement *up_,
                                                 const functionDeclStatement &other) :
      blockStatement(up_, other),
      function((function_t&) other.function.clone()) {
      addArgumentsToScope();
    }

    statement_t& functionDeclStatement::clone_(blockStatement *up_) const {
      functionDeclStatement *smnt = new functionDeclStatement(up_, *this);
      if (up_) {
        smnt->addFunctionToParentScope();
      }
      return *smnt;
    }

    int functionDeclStatement::type() const {
      return statementType::functionDecl;
    }

    std::string functionDeclStatement::statementName() const {
      return "function declaration";
    }

    bool functionDeclStatement::addFunctionToParentScope() {
      if (up && !up->addToScope(function)) {
        return false;
      }
      return true;
    }

    void functionDeclStatement::addArgumentsToScope() {
      const int count = (int) function.args.size();
      for (int i = 0; i < count; ++i) {
        addToScope(*(function.args[i]));
      }
    }

    void functionDeclStatement::print(printer &pout) const {
      // Double newlines to make it look cleaner
      pout.printNewlines(2);

      pout.printStartIndentation();
      function.printDeclaration(pout);
      pout << ' ';
      blockStatement::print(pout);

      // Double newlines to make it look cleaner
      pout.printNewlines(2);
    }
  }
}
