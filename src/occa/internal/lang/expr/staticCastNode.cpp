#include <occa/internal/lang/expr/staticCastNode.hpp>

namespace occa {
  namespace lang {
    staticCastNode::staticCastNode(token_t *token_,
                                   const vartype_t &valueType_,
                                   const exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()) {}

    staticCastNode::staticCastNode(const staticCastNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value->clone()) {}

    staticCastNode::~staticCastNode() {
      delete value;
    }

    udim_t staticCastNode::type() const {
      return exprNodeType::staticCast;
    }

    exprNode* staticCastNode::endNode() {
      return value->endNode();
    }

    exprNode* staticCastNode::clone() const {
      return new staticCastNode(token, valueType, *value);
    }

    void staticCastNode::pushChildNodes(exprNodeVector &children) {
      children.push_back(value);
    }

    bool staticCastNode::safeReplaceExprNode(exprNode *currentNode, exprNode *newNode) {
      if (currentNode == value) {
        delete value;
        value = newNode;
        return true;
      }

      return false;
    }

    void staticCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "static_cast<" << valueType << ">("
           << *value << ')';
    }

    void staticCastNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << valueType;
      io::stderr << "] (staticCast)\n";
      value->childDebugPrint(prefix);
    }
  }
}
