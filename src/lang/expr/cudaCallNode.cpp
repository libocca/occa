#include <occa/lang/expr/cudaCallNode.hpp>

namespace occa {
  namespace lang {
    cudaCallNode::cudaCallNode(token_t *token_,
                               const exprNode &value_,
                               const exprNode &blocks_,
                               const exprNode &threads_) :
      exprNode(token_),
      value(value_.clone()),
      blocks(blocks_.clone()),
      threads(threads_.clone()) {}

    cudaCallNode::cudaCallNode(const cudaCallNode &node) :
      exprNode(node.token),
      value(node.value->clone()),
      blocks(node.blocks->clone()),
      threads(node.threads->clone()) {}

    cudaCallNode::~cudaCallNode() {
      delete value;
      delete blocks;
      delete threads;
    }

    udim_t cudaCallNode::type() const {
      return exprNodeType::cudaCall;
    }

    exprNode* cudaCallNode::startNode() {
      return value->startNode();
    }

    exprNode* cudaCallNode::endNode() {
      return threads->endNode();
    }

    exprNode* cudaCallNode::clone() const {
      return new cudaCallNode(token,
                              *value,
                              *blocks,
                              *threads);
    }

    void cudaCallNode::setChildren(exprNodeRefVector &children) {
      children.push_back(&value);
      children.push_back(&blocks);
      children.push_back(&threads);
    }

    void cudaCallNode::print(printer &pout) const {
      pout << *value
           << "<<<"
           << *blocks << ", " << *threads
           << ">>>";
    }

    void cudaCallNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[<<<...>>>";
      io::stderr << "] (cudaCall)\n";
      value->childDebugPrint(prefix);
      blocks->childDebugPrint(prefix);
      threads->childDebugPrint(prefix);
    }
  }
}
