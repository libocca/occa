#include <occa/internal/lang/expr/constCastNode.hpp>

namespace occa {
  namespace lang {
    constCastNode::constCastNode(token_t *token_,
                                 const vartype_t &valueType_,
                                 const exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()) {}

    constCastNode::constCastNode(const constCastNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value->clone()) {}

    constCastNode::~constCastNode() {
      delete value;
    }

    udim_t constCastNode::type() const {
      return exprNodeType::constCast;
    }

    exprNode* constCastNode::endNode() {
      return value->endNode();
    }

    exprNode* constCastNode::clone() const {
      return new constCastNode(token, valueType, *value);
    }

    void constCastNode::pushChildNodes(exprNodeVector &children) {
      children.push_back(value);
    }

    bool constCastNode::safeReplaceExprNode(exprNode *currentNode, exprNode *newNode) {
      if (currentNode == value) {
        delete value;
        value = newNode;
        return true;
      }

      return false;
    }

    void constCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "const_cast<" << valueType << ">("
           << *value << ')';
    }

    void constCastNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << valueType;
      io::stderr << "] (constCast)\n";
      value->childDebugPrint(prefix);
    }
  }
}
