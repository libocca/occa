#include <occa/internal/lang/statement/functionStatement.hpp>
#include <occa/internal/lang/expr/functionNode.hpp>

namespace occa {
  namespace lang {
    functionStatement::functionStatement(blockStatement *up_,
                                         function_t &function_) :
      statement_t(up_, function_.source),
      funcNode(new functionNode(function_.source,
                                function_)) {}

    functionStatement::functionStatement(blockStatement *up_,
                                         const functionStatement &other) :
      statement_t(up_, other),
      funcNode((functionNode*) other.funcNode->clone()) {}

    functionStatement::~functionStatement() {
      delete funcNode;
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

    function_t& functionStatement::function() {
      return funcNode->value;
    }

    const function_t& functionStatement::function() const {
      return funcNode->value;
    }

    exprNodeArray functionStatement::getDirectExprNodes() {
      exprNodeArray arr;

      arr.push({this, funcNode});

      return arr;
    }

    void functionStatement::safeReplaceExprNode(exprNode *currentNode, exprNode *newNode) {
      if (funcNode == currentNode) {
        delete funcNode;
        funcNode = (functionNode*) exprNode::clone(newNode);
      }
    }

    void functionStatement::print(printer &pout) const {
      pout.printStartIndentation();
      function().printDeclaration(pout);
      // Double newlines to make it look cleaner
      pout << ";\n\n";
    }
  }
}
