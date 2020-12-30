#include <occa/internal/lang/expr/emptyNode.hpp>

namespace occa {
  namespace lang {
    const emptyNode noExprNode;

    emptyNode::emptyNode() :
      exprNode(NULL) {}

    emptyNode::~emptyNode() {}

    udim_t emptyNode::type() const {
      return exprNodeType::empty;
    }

    exprNode* emptyNode::clone() const {
      return new emptyNode();
    }

    void emptyNode::print(printer &pout) const {}

    void emptyNode::debugPrint(const std::string &prefix) const {
      io::stderr << prefix << "|\n"
                << prefix << "|---o\n"
                << prefix << '\n';
    }
  }
}
