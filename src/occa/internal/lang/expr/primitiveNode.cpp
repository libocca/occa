#include <occa/internal/lang/expr/primitiveNode.hpp>

namespace occa {
  namespace lang {
    primitiveNode::primitiveNode(token_t *token_,
                                 primitive value_) :
      exprNode(token_),
      value(value_) {}

    primitiveNode::primitiveNode(const primitiveNode &node) :
      exprNode(node.token),
      value(node.value) {}

    primitiveNode::~primitiveNode() {}

    udim_t primitiveNode::type() const {
      return exprNodeType::primitive;
    }

    exprNode* primitiveNode::clone() const {
      return new primitiveNode(token, value);
    }

    bool primitiveNode::canEvaluate() const {
      return true;
    }

    primitive primitiveNode::evaluate() const {
      return value;
    }

    void primitiveNode::print(printer &pout) const {
      pout << value.toString();
    }

    void primitiveNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << (*this);
      io::stderr << "] (primitive)\n";
    }
  }
}
