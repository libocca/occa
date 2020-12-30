#include <occa/internal/lang/expr/identifierNode.hpp>

namespace occa {
  namespace lang {
    identifierNode::identifierNode(token_t *token_,
                                   const std::string &value_) :
      exprNode(token_),
      value(value_) {}

    identifierNode::identifierNode(const identifierNode &node) :
      exprNode(node.token),
      value(node.value) {}

    identifierNode::~identifierNode() {}

    udim_t identifierNode::type() const {
      return exprNodeType::identifier;
    }

    exprNode* identifierNode::clone() const {
      return new identifierNode(token, value);
    }

    void identifierNode::print(printer &pout) const {
      pout << value;
    }

    void identifierNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << '\n'
                << prefix << "|---[";
      pout << (*this);
      io::stderr << "] (identifier)\n";
    }
  }
}
