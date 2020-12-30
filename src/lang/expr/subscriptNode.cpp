#include <occa/internal/lang/expr/subscriptNode.hpp>

namespace occa {
  namespace lang {
    subscriptNode::subscriptNode(token_t *token_,
                                 const exprNode &value_,
                                 const exprNode &index_) :
      exprNode(token_),
      value(value_.clone()),
      index(index_.clone()) {}

    subscriptNode::subscriptNode(const subscriptNode &node) :
      exprNode(node.token),
      value(node.value->clone()),
      index(node.index->clone()) {}

    subscriptNode::~subscriptNode() {
      delete value;
      delete index;
    }

    udim_t subscriptNode::type() const {
      return exprNodeType::subscript;
    }

    exprNode* subscriptNode::clone() const {
      return new subscriptNode(token, *value, *index);
    }

    exprNode* subscriptNode::startNode() {
      return value->startNode();
    }

    exprNode* subscriptNode::endNode() {
      return index->endNode();
    }

    void subscriptNode::pushChildNodes(exprNodeVector &children) {
      children.push_back(value);
      children.push_back(index);
    }

    bool subscriptNode::safeReplaceExprNode(exprNode *currentNode, exprNode *newNode) {
      if (currentNode == value) {
        delete value;
        value = newNode;
        return true;
      }

      if (currentNode == index) {
        delete index;
        index = newNode;
        return true;
      }

      return false;
    }

    variable_t* subscriptNode::getVariable() {
      return value->getVariable();
    }

    void subscriptNode::print(printer &pout) const {
      pout << *value
           << '[' << *index << ']';
    }

    void subscriptNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << *index;
      io::stderr << "] (subscript)\n";
      value->childDebugPrint(prefix);
    }
  }
}
