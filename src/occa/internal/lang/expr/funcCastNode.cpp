#include <occa/internal/lang/expr/funcCastNode.hpp>

namespace occa {
  namespace lang {
    funcCastNode::funcCastNode(token_t *token_,
                               const vartype_t &valueType_,
                               const exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()) {}

    funcCastNode::funcCastNode(const funcCastNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value->clone()) {}

    funcCastNode::~funcCastNode() {
      delete value;
    }

    udim_t funcCastNode::type() const {
      return exprNodeType::funcCast;
    }

    exprNode* funcCastNode::startNode() {
      return value->startNode();
    }

    exprNode* funcCastNode::endNode() {
      return value->endNode();
    }

    exprNode* funcCastNode::clone() const {
      return new funcCastNode(token, valueType, *value);
    }

    void funcCastNode::pushChildNodes(exprNodeVector &children) {
      children.push_back(value);
    }

    bool funcCastNode::safeReplaceExprNode(exprNode *currentNode, exprNode *newNode) {
      if (currentNode == value) {
        delete value;
        value = newNode;
        return true;
      }

      return false;
    }

    void funcCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << valueType << '(' << *value << ')';
    }

    void funcCastNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << valueType;
      io::stderr << "] (funcCast)\n";
      value->childDebugPrint(prefix);
    }
  }
}
