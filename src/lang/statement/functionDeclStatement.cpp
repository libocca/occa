#include <occa/internal/lang/statement/functionDeclStatement.hpp>
#include <occa/internal/lang/expr/functionNode.hpp>
#include <occa/internal/lang/variable.hpp>

namespace occa {
  namespace lang {
    functionDeclStatement::functionDeclStatement(blockStatement *up_,
                                                 function_t &function_) :
      blockStatement(up_, function_.source),
      funcNode(new functionNode(function_.source,
                                function_)) {

      addFunctionToParentScope();
      addArgumentsToScope();
    }

    functionDeclStatement::functionDeclStatement(blockStatement *up_,
                                                 const functionDeclStatement &other) :
        blockStatement(up_, other.source),
        funcNode(new functionNode(other.function().source,
                                  (function_t&) other.function().clone())) {
      // Add our cloned variables to scope before copying from other
      // This way the keywords won't be copied first
      addFunctionToParentScope();
      addArgumentsToScope();

      copyFrom(other);

      replaceFunction(other.function(), function());
    }

    functionDeclStatement::~functionDeclStatement() {
      delete funcNode;
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

    function_t& functionDeclStatement::function() {
      return funcNode->value;
    }

    const function_t& functionDeclStatement::function() const {
      return funcNode->value;
    }

    bool functionDeclStatement::addFunctionToParentScope() {
      if (up && !up->addToScope(function())) {
        return false;
      }
      return true;
    }

    void functionDeclStatement::addArgumentsToScope() {
      for (auto arg : function().args) {
        addToScope(*arg);
      }
    }

    exprNodeArray functionDeclStatement::getDirectExprNodes() {
      exprNodeArray arr;

      if (funcNode) {
        arr.push({this, (exprNode*) funcNode});
      }

      return arr;
    }

    void functionDeclStatement::safeReplaceExprNode(exprNode *currentNode, exprNode *newNode) {
      if (funcNode == currentNode) {
        delete funcNode;
        funcNode = (functionNode*) exprNode::clone(newNode);
      }
    }

    void functionDeclStatement::print(printer &pout) const {
      // Double newlines to make it look cleaner
      pout.printNewlines(2);

      pout.printStartIndentation();
      function().printDeclaration(pout);
      pout << ' ';
      blockStatement::print(pout);

      // Double newlines to make it look cleaner
      pout.printNewlines(2);
    }
  }
}
