#include <occa/lang/expr/dynamicCastNode.hpp>

namespace occa {
  namespace lang {
    dynamicCastNode::dynamicCastNode(token_t *token_,
                                     const vartype_t &valueType_,
                                     const exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()) {}

    dynamicCastNode::dynamicCastNode(const dynamicCastNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value->clone()) {}

    dynamicCastNode::~dynamicCastNode() {
      delete value;
    }

    udim_t dynamicCastNode::type() const {
      return exprNodeType::dynamicCast;
    }

    exprNode* dynamicCastNode::endNode() {
      return value->endNode();
    }

    exprNode* dynamicCastNode::clone() const {
      return new dynamicCastNode(token, valueType, *value);
    }

    void dynamicCastNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&value);
    }

    void dynamicCastNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "dynamic_cast<" << valueType << ">("
           << *value << ')';
    }

    void dynamicCastNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << valueType;
      io::stderr << "] (dynamicCast)\n";
      value->childDebugPrint(prefix);
    }
  }
}
