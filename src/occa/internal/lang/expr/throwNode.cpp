#include <occa/internal/lang/expr/throwNode.hpp>

namespace occa {
  namespace lang {
    throwNode::throwNode(token_t *token_,
                         const exprNode &value_) :
      exprNode(token_),
      value(value_.clone()) {}

    throwNode::throwNode(const throwNode &node) :
      exprNode(node.token),
      value(node.value->clone()) {}

    throwNode::~throwNode() {
      delete value;
    }

    udim_t throwNode::type() const {
      return exprNodeType::throw_;
    }

    exprNode* throwNode::clone() const {
      return new throwNode(token, *value);
    }

    exprNode* throwNode::endNode() {
      return value->endNode();
    }

    void throwNode::pushChildNodes(exprNodeVector &children) {
      children.push_back(value);
    }

    bool throwNode::safeReplaceExprNode(exprNode *currentNode, exprNode *newNode) {
      if (currentNode == value) {
        delete value;
        value = newNode;
        return true;
      }

      return false;
    }

    exprNode* throwNode::wrapInParentheses() {
      return new parenthesesNode(token, *this);
    }

    void throwNode::print(printer &pout) const {
      pout << "throw";
      if (value->type() != exprNodeType::empty) {
        pout << ' ' << *value;
      }
    }

    void throwNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|\n"
                << prefix << "|---[";
      pout << *value;
      io::stderr << "] (throw)\n";
    }
  }
}
