#include <occa/internal/lang/expr/sizeofNode.hpp>

namespace occa {
  namespace lang {
    sizeofNode::sizeofNode(token_t *token_,
                           const exprNode &value_) :
      exprNode(token_),
      value(value_.clone()) {}

    sizeofNode::sizeofNode(const sizeofNode &node) :
      exprNode(node.token),
      value(node.value->clone()) {}

    sizeofNode::~sizeofNode() {
      delete value;
    }

    udim_t sizeofNode::type() const {
      return exprNodeType::sizeof_;
    }

    exprNode* sizeofNode::endNode() {
      return value->endNode();
    }

    exprNode* sizeofNode::clone() const {
      return new sizeofNode(token, *value);
    }

    bool sizeofNode::canEvaluate() const {
      return value->canEvaluate();
    }

    primitive sizeofNode::evaluate() const {
      return value->evaluate().sizeof_();
    }

    void sizeofNode::pushChildNodes(exprNodeVector &children) {
      children.push_back(value);
    }

    bool sizeofNode::safeReplaceExprNode(exprNode *currentNode, exprNode *newNode) {
      if (currentNode == value) {
        delete value;
        value = newNode;
        return true;
      }

      return false;
    }

    void sizeofNode::print(printer &pout) const {
      pout << "sizeof(" << *value << ')';
    }

    void sizeofNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << '\n'
                << prefix << "|---[";
      pout << *value;
      io::stderr << "] (sizeof)\n";
    }
  }
}
