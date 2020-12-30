#include <occa/internal/lang/expr/rightUnaryOpNode.hpp>

namespace occa {
  namespace lang {
    rightUnaryOpNode::rightUnaryOpNode(token_t *token_,
                                       const unaryOperator_t &op_,
                                       const exprNode &value_) :
      exprOpNode(token_, op_),
      value(value_.clone()) {}

    rightUnaryOpNode::rightUnaryOpNode(const rightUnaryOpNode &node) :
      exprOpNode(node.token, node.op),
      value(node.value->clone()) {}

    rightUnaryOpNode::~rightUnaryOpNode() {
      delete value;
    }

    udim_t rightUnaryOpNode::type() const {
      return exprNodeType::rightUnary;
    }

    exprNode* rightUnaryOpNode::clone() const {
      return new rightUnaryOpNode(token,
                                  (const unaryOperator_t&) op,
                                  *value);
    }

    bool rightUnaryOpNode::canEvaluate() const {
      return value->canEvaluate();
    }

    primitive rightUnaryOpNode::evaluate() const {
      primitive pValue = value->evaluate();
      return ((unaryOperator_t&) op)(pValue);
    }

    exprNode* rightUnaryOpNode::startNode() {
      return value->startNode();
    }

    void rightUnaryOpNode::pushChildNodes(exprNodeVector &children) {
      children.push_back(value);
    }

    bool rightUnaryOpNode::safeReplaceExprNode(exprNode *currentNode, exprNode *newNode) {
      if (currentNode == value) {
        delete value;
        value = newNode;
        return true;
      }

      return false;
    }

    variable_t* rightUnaryOpNode::getVariable() {
      return value->getVariable();
    }

    void rightUnaryOpNode::print(printer &pout) const {
      pout << *value << op;
    }

    void rightUnaryOpNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << op;
      io::stderr << "] (rightUnary)\n";
      value->childDebugPrint(prefix);
    }
  }
}
