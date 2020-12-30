#include <occa/internal/lang/expr/ternaryOpNode.hpp>

namespace occa {
  namespace lang {
    ternaryOpNode::ternaryOpNode(const exprNode &checkValue_,
                                 const exprNode &trueValue_,
                                 const exprNode &falseValue_) :
      exprOpNode(checkValue_.token, op::ternary),
      checkValue(checkValue_.clone()),
      trueValue(trueValue_.clone()),
      falseValue(falseValue_.clone()) {}

    ternaryOpNode::ternaryOpNode(const ternaryOpNode &node) :
      exprOpNode(node.token, op::ternary),
      checkValue(node.checkValue->clone()),
      trueValue(node.trueValue->clone()),
      falseValue(node.falseValue->clone()) {}

    ternaryOpNode::~ternaryOpNode() {
      delete checkValue;
      delete trueValue;
      delete falseValue;
    }

    udim_t ternaryOpNode::type() const {
      return exprNodeType::ternary;
    }

    opType_t ternaryOpNode::opType() const {
      return operatorType::ternary;
    }

    exprNode* ternaryOpNode::clone() const {
      return new ternaryOpNode(*checkValue,
                               *trueValue,
                               *falseValue);
    }

    bool ternaryOpNode::canEvaluate() const {
      return (checkValue->canEvaluate() &&
              trueValue->canEvaluate()  &&
              falseValue->canEvaluate());
    }

    primitive ternaryOpNode::evaluate() const {
      if ((bool) checkValue->evaluate()) {
        return trueValue->evaluate();
      }
      return falseValue->evaluate();
    }

    exprNode* ternaryOpNode::startNode() {
      return checkValue->startNode();
    }

    exprNode* ternaryOpNode::endNode() {
      return falseValue->endNode();
    }

    void ternaryOpNode::pushChildNodes(exprNodeVector &children) {
      children.push_back(checkValue);
      children.push_back(trueValue);
      children.push_back(falseValue);
    }

    bool ternaryOpNode::safeReplaceExprNode(exprNode *currentNode, exprNode *newNode) {
      if (currentNode == checkValue) {
        delete checkValue;
        checkValue = newNode;
        return true;
      }

      if (currentNode == trueValue) {
        delete trueValue;
        trueValue = newNode;
        return true;
      }

      if (currentNode == falseValue) {
        delete falseValue;
        falseValue = newNode;
        return true;
      }

      return false;
    }

    void ternaryOpNode::print(printer &pout) const {
      pout << *checkValue
           << " ? " << *trueValue
           << " : " << *falseValue;
    }

    void ternaryOpNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[?:] (ternary)\n";
      checkValue->childDebugPrint(prefix);
      trueValue->childDebugPrint(prefix);
      falseValue->childDebugPrint(prefix);
    }
  }
}
