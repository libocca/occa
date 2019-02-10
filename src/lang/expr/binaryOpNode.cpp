#include <occa/lang/expr/binaryOpNode.hpp>

namespace occa {
  namespace lang {
    binaryOpNode::binaryOpNode(token_t *token_,
                               const binaryOperator_t &op_,
                               const exprNode &leftValue_,
                               const exprNode &rightValue_) :
      exprOpNode(token_, op_),
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

    void binaryOpNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&leftValue);
      children.push_back(&rightValue);
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
