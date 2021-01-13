#include <occa/internal/lang/expr/leftUnaryOpNode.hpp>

namespace occa {
  namespace lang {
    leftUnaryOpNode::leftUnaryOpNode(token_t *token_,
                                     const unaryOperator_t &op_,
                                     const exprNode &value_) :
      exprOpNode(token_, op_),
      value(value_.clone()) {}

    leftUnaryOpNode::leftUnaryOpNode(const leftUnaryOpNode &node) :
      exprOpNode(node.token, node.op),
      value(node.value->clone()) {}

    leftUnaryOpNode::~leftUnaryOpNode() {
      delete value;
    }

    udim_t leftUnaryOpNode::type() const {
      return exprNodeType::leftUnary;
    }

    exprNode* leftUnaryOpNode::clone() const {
      return new leftUnaryOpNode(token,
                                 (const unaryOperator_t&) op,
                                 *value);
    }

    bool leftUnaryOpNode::canEvaluate() const {
      if (op.opType & (operatorType::dereference |
                       operatorType::address)) {
        return false;
      }
      return value->canEvaluate();
    }

    primitive leftUnaryOpNode::evaluate() const {
      primitive pValue = value->evaluate();
      return ((unaryOperator_t&) op)(pValue);
    }

    exprNode* leftUnaryOpNode::endNode() {
      return value->endNode();
    }

    void leftUnaryOpNode::pushChildNodes(exprNodeVector &children) {
      children.push_back(value);
    }

    bool leftUnaryOpNode::safeReplaceExprNode(exprNode *currentNode, exprNode *newNode) {
      if (currentNode == value) {
        delete value;
        value = newNode;
        return true;
      }

      return false;
    }

    variable_t* leftUnaryOpNode::getVariable() {
      return value->getVariable();
    }

    void leftUnaryOpNode::print(printer &pout) const {
      pout << op << *value;
    }

    void leftUnaryOpNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << op;
      io::stderr << "] (leftUnary)\n";
      value->childDebugPrint(prefix);
    }
  }
}
