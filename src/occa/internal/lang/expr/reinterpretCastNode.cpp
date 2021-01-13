#include <occa/internal/lang/expr/reinterpretCastNode.hpp>

namespace occa {
  namespace lang {
    reinterpretCastNode::reinterpretCastNode(token_t *token_,
                                             const vartype_t &valueType_,
                                             const exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()) {}

    reinterpretCastNode::reinterpretCastNode(const reinterpretCastNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value->clone()) {}

    reinterpretCastNode::~reinterpretCastNode() {
      delete value;
    }

    udim_t reinterpretCastNode::type() const {
      return exprNodeType::reinterpretCast;
    }

    exprNode* reinterpretCastNode::endNode() {
      return value->endNode();
    }

    exprNode* reinterpretCastNode::clone() const {
      return new reinterpretCastNode(token, valueType, *value);
    }

    void reinterpretCastNode::pushChildNodes(exprNodeVector &children) {
      children.push_back(value);
    }

    bool reinterpretCastNode::safeReplaceExprNode(exprNode *currentNode, exprNode *newNode) {
      if (currentNode == value) {
        delete value;
        value = newNode;
        return true;
      }

      return false;
    }

    void reinterpretCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "reinterpret_cast<" << valueType << ">("
           << *value << ')';
    }

    void reinterpretCastNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << valueType;
      io::stderr << "] (reinterpretCast)\n";
      value->childDebugPrint(prefix);
    }
  }
}
