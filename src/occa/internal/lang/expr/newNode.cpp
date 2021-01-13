#include <occa/internal/lang/expr/newNode.hpp>

namespace occa {
  namespace lang {
    newNode::newNode(token_t *token_,
                     const vartype_t &valueType_,
                     const exprNode &value_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()),
      size(noExprNode.clone()) {}

    newNode::newNode(token_t *token_,
                     const vartype_t &valueType_,
                     const exprNode &value_,
                     const exprNode &size_) :
      exprNode(token_),
      valueType(valueType_),
      value(value_.clone()),
      size(size_.clone()) {}

    newNode::newNode(const newNode &node) :
      exprNode(node.token),
      valueType(node.valueType),
      value(node.value->clone()),
      size(node.size->clone()) {}

    newNode::~newNode() {
      delete value;
      delete size;
    }

    udim_t newNode::type() const {
      return exprNodeType::new_;
    }

    exprNode* newNode::clone() const {
      return new newNode(token, valueType, *value, *size);
    }

    exprNode* newNode::endNode() {
      return (size ? size : value)->endNode();
    }

    void newNode::pushChildNodes(exprNodeVector &children) {
      children.push_back(value);
      children.push_back(size);
    }

    bool newNode::safeReplaceExprNode(exprNode *currentNode, exprNode *newNode_) {
      if (currentNode == value) {
        delete value;
        value = newNode_;
        return true;
      }

      if (currentNode == size) {
        delete size;
        size = newNode_;
        return true;
      }

      return false;
    }

    exprNode* newNode::wrapInParentheses() {
      return new parenthesesNode(token, *this);
    }

    void newNode::print(printer &pout) const {
      // TODO: Print type without qualifiers
      //       Also convert [] to *
      pout << "new " << valueType << *value;
      if (size->type() != exprNodeType::empty) {
        pout << '[' << *size << ']';
      }
    }

    void newNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << valueType;
      io::stderr << "] (new)\n";
      value->childDebugPrint(prefix);
      size->childDebugPrint(prefix);
    }
  }
}
