#include <occa/lang/statement/functionDeclStatement.hpp>

namespace occa {
  namespace lang {
    functionDeclStatement::functionDeclStatement(blockStatement *up_,
                                                 function_t &function_) :
      blockStatement(up_, function_.source),
      function(function_) {}

    functionDeclStatement::functionDeclStatement(blockStatement *up_,
                                                 const functionDeclStatement &other) :
      blockStatement(up_, other),
      function((function_t&) other.function.clone()) {
      updateScope(true);
    }

    statement_t& functionDeclStatement::clone_(blockStatement *up_) const {
      return *(new functionDeclStatement(up_, *this));
    }

    int functionDeclStatement::type() const {
      return statementType::functionDecl;
    }

    std::string functionDeclStatement::statementName() const {
      return "function declaration";
    }

    bool functionDeclStatement::updateScope(const bool force) {
      if (up && !up->addToScope(function, force)) {
        return false;
      }
      addArgumentsToScope(force);
      return true;
    }

    void functionDeclStatement::addArgumentsToScope(const bool force) {
      const int count = (int) function.args.size();
      for (int i = 0; i < count; ++i) {
        addToScope(*(function.args[i]),
                   force);
      }
    }

    void functionDeclStatement::print(printer &pout) const {
      // Double newlines to make it look cleaner
      pout << '\n';
      pout.printStartIndentation();
      function.printDeclaration(pout);
      pout << ' ';
      blockStatement::print(pout);
    }
  }
}
