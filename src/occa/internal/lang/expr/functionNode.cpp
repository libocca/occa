#include <occa/internal/lang/expr/functionNode.hpp>

namespace occa {
  namespace lang {
    functionNode::functionNode(token_t *token_,
                               function_t &value_) :
      exprNode(token_),
      value(value_) {}

    functionNode::functionNode(const functionNode &node) :
      exprNode(node.token),
      value(node.value) {}

    functionNode::~functionNode() {}

    udim_t functionNode::type() const {
      return exprNodeType::function;
    }

    exprNode* functionNode::clone() const {
      return new functionNode(token, value);
    }

    bool functionNode::hasAttribute(const std::string &attr) const {
      return value.hasAttribute(attr);
    }

    void functionNode::print(printer &pout) const {
      pout << value;
    }

    void functionNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << (*this);
      io::stderr << "] (function)\n";
    }
  }
}
