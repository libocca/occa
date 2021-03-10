#include <occa/internal/lang/expr/charNode.hpp>

namespace occa {
  namespace lang {
    charNode::charNode(token_t *token_,
                       const std::string &value_) :
      exprNode(token_),
      value(value_) {}

    charNode::charNode(const charNode &node) :
      exprNode(node.token),
      value(node.value) {}

    charNode::~charNode() {}

    udim_t charNode::type() const {
      return exprNodeType::char_;
    }

    exprNode* charNode::clone() const {
      return new charNode(token, value);
    }

    void charNode::print(printer &pout) const {
      pout << '\'' << escape(value, '\'') << '\'';
    }

    void charNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << '\n'
                << prefix << "|---[";
      pout << (*this);
      io::stderr << "] (char)\n";
    }
  }
}
