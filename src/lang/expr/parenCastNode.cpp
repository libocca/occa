#include <occa/lang/expr/parenCastNode.hpp>

namespace occa {
  namespace lang {
    parenCastNode::parenCastNode(token_t *token_,
                                 const vartype_t &valueType_,
                                 const exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()) {}

    parenCastNode::parenCastNode(const parenCastNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value->clone()) {}

    parenCastNode::~parenCastNode() {
      delete value;
    }

    udim_t parenCastNode::type() const {
      return exprNodeType::parenCast;
    }

    exprNode* parenCastNode::startNode() {
      return value->startNode();
    }

    exprNode* parenCastNode::endNode() {
      return value->endNode();
    }

    exprNode* parenCastNode::clone() const {
      return new parenCastNode(token, valueType, *value);
    }

    void parenCastNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&value);
    }

    exprNode* parenCastNode::wrapInParentheses() {
      return new parenthesesNode(token, *this);
    }

    void parenCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << '(' << valueType << ") " << *value;
    }

    void parenCastNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << valueType;
      io::stderr << "] (parenCast)\n";
      value->childDebugPrint(prefix);
    }
  }
}
