#include <occa/internal/lang/variable.hpp>
#include <occa/internal/lang/expr/variableNode.hpp>

namespace occa {
  namespace lang {
    variableNode::variableNode(token_t *token_,
                               variable_t &value_) :
      exprNode(token_),
      value(value_) {}

    variableNode::variableNode(const variableNode &node) :
      exprNode(node.token),
      value(node.value) {}

    variableNode::~variableNode() {}

    udim_t variableNode::type() const {
      return exprNodeType::variable;
    }

    exprNode* variableNode::clone() const {
      return new variableNode(token, value);
    }

    bool variableNode::hasAttribute(const std::string &attr) const {
      return value.hasAttribute(attr);
    }

    variable_t* variableNode::getVariable() {
      return &value;
    }

    void variableNode::print(printer &pout) const {
      pout << value;
    }

    void variableNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                 << prefix << "|---[";
      pout << (*this);
      io::stderr << "] (variable)\n";
    }
  }
}
