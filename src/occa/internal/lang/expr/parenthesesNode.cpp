#include <occa/internal/lang/expr/parenthesesNode.hpp>

namespace occa {
  namespace lang {
    parenthesesNode::parenthesesNode(token_t *token_,
                                     const exprNode &value_) :
      exprNode(token_),
      value(value_.clone()) {}

    parenthesesNode::parenthesesNode(const parenthesesNode &node) :
      exprNode(node.token),
      value(node.value->clone()) {}

    parenthesesNode::~parenthesesNode() {
      delete value;
    }

    udim_t parenthesesNode::type() const {
      return exprNodeType::parentheses;
    }

    exprNode* parenthesesNode::startNode() {
      return value->startNode();
    }

    exprNode* parenthesesNode::endNode() {
      return value->endNode();
    }

    exprNode* parenthesesNode::clone() const {
      return new parenthesesNode(token, *value);
    }

    bool parenthesesNode::canEvaluate() const {
      return value->canEvaluate();
    }

    primitive parenthesesNode::evaluate() const {
      return value->evaluate();
    }

    void parenthesesNode::pushChildNodes(exprNodeVector &children) {
      children.push_back(value);
    }

    bool parenthesesNode::safeReplaceExprNode(exprNode *currentNode, exprNode *newNode) {
      if (currentNode == value) {
        delete value;
        value = newNode;
        return true;
      }

      return false;
    }

    variable_t* parenthesesNode::getVariable() {
      return value->getVariable();
    }

    void parenthesesNode::print(printer &pout) const {
      pout << '(' << *value << ')';
    }

    void parenthesesNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[()] (parentheses)\n";
      value->childDebugPrint(prefix);
    }
  }
}
