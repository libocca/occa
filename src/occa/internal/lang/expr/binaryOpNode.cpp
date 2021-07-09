#include <occa/internal/lang/expr/binaryOpNode.hpp>

namespace occa {
  namespace lang {
    binaryOpNode::binaryOpNode(token_t *token_,
                               const binaryOperator_t &op_,
                               const exprNode &leftValue_,
                               const exprNode &rightValue_) :
      exprOpNode(token_, op_),
      leftValue(leftValue_.clone()),
      rightValue(rightValue_.clone()) {}

      binaryOpNode::binaryOpNode(const binaryOperator_t &op_,
                           const exprNode &leftValue_,
                           const exprNode &rightValue_) :
      exprOpNode(leftValue_.token, op_),
      leftValue(leftValue_.clone()),
      rightValue(rightValue_.clone()) {}

    binaryOpNode::binaryOpNode(const binaryOpNode &node) :
      exprOpNode(node.token, node.op),
      leftValue(node.leftValue->clone()),
      rightValue(node.rightValue->clone()) {}

    binaryOpNode::~binaryOpNode() {
      delete leftValue;
      delete rightValue;
    }

    udim_t binaryOpNode::type() const {
      return exprNodeType::binary;
    }

    exprNode* binaryOpNode::clone() const {
      return new binaryOpNode(token,
                              (const binaryOperator_t&) op,
                              *leftValue,
                              *rightValue);
    }

    bool binaryOpNode::canEvaluate() const {
      if (op.opType & (operatorType::scope     |
                       operatorType::dot       |
                       operatorType::dotStar   |
                       operatorType::arrow     |
                       operatorType::arrowStar)) {
        return false;
      }
      return (leftValue->canEvaluate() &&
              rightValue->canEvaluate());
    }

    primitive binaryOpNode::evaluate() const {
      primitive pLeft  = leftValue->evaluate();
      primitive pRight = rightValue->evaluate();
      return ((binaryOperator_t&) op)(pLeft, pRight);
    }

    exprNode* binaryOpNode::startNode() {
      return leftValue->startNode();
    }

    exprNode* binaryOpNode::endNode() {
      return rightValue->endNode();
    }

    void binaryOpNode::pushChildNodes(exprNodeVector &children) {
      children.push_back(leftValue);
      children.push_back(rightValue);
    }

    bool binaryOpNode::safeReplaceExprNode(exprNode *currentNode, exprNode *newNode) {
      if (currentNode == leftValue) {
        delete leftValue;
        leftValue = newNode;
        return true;
      }

      if (currentNode == rightValue) {
        delete rightValue;
        rightValue = newNode;
        return true;
      }

      return false;
    }

    variable_t* binaryOpNode::getVariable() {
      return leftValue->getVariable();
    }

    void binaryOpNode::print(printer &pout) const {
      if (op.opType & (operatorType::scope |
                       operatorType::dot |
                       operatorType::dotStar |
                       operatorType::arrow |
                       operatorType::arrowStar)) {
        pout << *leftValue << op << *rightValue;
      }
      else if (op.opType & operatorType::comma) {
        pout << *leftValue << ", " << *rightValue;
      } else {
        pout << *leftValue << ' ' << op << ' ' << *rightValue;
      }
    }

    void binaryOpNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << op;
      io::stderr << "] (binary)\n";
      leftValue->childDebugPrint(prefix);
      rightValue->childDebugPrint(prefix);
    }
  }
}
