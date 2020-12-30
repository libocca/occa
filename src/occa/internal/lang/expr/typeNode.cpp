#include <occa/internal/lang/expr/typeNode.hpp>

namespace occa {
  namespace lang {
    typeNode::typeNode(token_t *token_,
                       type_t &value_) :
      exprNode(token_),
      value(value_) {}

    typeNode::typeNode(const typeNode &node) :
      exprNode(node.token),
      value(node.value) {}

    typeNode::~typeNode() {}

    udim_t typeNode::type() const {
      return exprNodeType::type;
    }

    exprNode* typeNode::clone() const {
      return new typeNode(token, value);
    }

    bool typeNode::hasAttribute(const std::string &attr) const {
      return value.hasAttribute(attr);
    }

    void typeNode::print(printer &pout) const {
      pout << value;
    }

    void typeNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << (*this);
      io::stderr << "] (type)\n";
    }
  }
}
