#include <occa/internal/lang/expr/vartypeNode.hpp>

namespace occa {
  namespace lang {
    vartypeNode::vartypeNode(token_t *token_,
                             const vartype_t &value_) :
      exprNode(token_),
      value(value_) {}

    vartypeNode::vartypeNode(const vartypeNode &node) :
      exprNode(node.token),
      value(node.value) {}

    vartypeNode::~vartypeNode() {}

    udim_t vartypeNode::type() const {
      return exprNodeType::vartype;
    }

    exprNode* vartypeNode::clone() const {
      return new vartypeNode(token, value);
    }

    bool vartypeNode::hasAttribute(const std::string &attr) const {
      return value.hasAttribute(attr);
    }

    void vartypeNode::print(printer &pout) const {
      pout << value;
    }

    void vartypeNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << (*this);
      io::stderr << "] (vartype)\n";
    }
  }
}
