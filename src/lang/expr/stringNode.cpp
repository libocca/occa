#include <occa/lang/expr/stringNode.hpp>

namespace occa {
  namespace lang {
    stringNode::stringNode(token_t *token_,
                           const std::string &value_) :
      exprNode(token_),
      value(value_) {}

    stringNode::stringNode(const stringNode &node) :
      exprNode(node.token),
      value(node.value) {}

    stringNode::~stringNode() {}

    udim_t stringNode::type() const {
      return exprNodeType::string;
    }

    exprNode* stringNode::clone() const {
      return new stringNode(token, value);
    }

    void stringNode::setChildren(exprNodeRefVector &children) {}

    void stringNode::print(printer &pout) const {
      pout << "\"" << escape(value, '"') << "\"";
    }

    void stringNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << (*this);
      io::stderr << "] (string)\n";
    }
  }
}
