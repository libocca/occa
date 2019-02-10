#include <occa/lang/expr/tupleNode.hpp>

namespace occa {
  namespace lang {
    tupleNode::tupleNode(token_t *token_,
                         const exprNodeVector &args_) :
      exprNode(token_) {
      cloneExprNodeVector(args, args_);
    }

    tupleNode::tupleNode(const tupleNode &node) :
      exprNode(node.token) {
      cloneExprNodeVector(args, node.args);
    }

    tupleNode::~tupleNode() {
      freeExprNodeVector(args);
    }

    udim_t tupleNode::type() const {
      return exprNodeType::tuple;
    }

    exprNode* tupleNode::startNode() {
      const int argCount = (int) args.size();
      return (argCount ? args[0]->startNode() : this);
    }

    exprNode* tupleNode::endNode() {
      const int argCount = (int) args.size();
      return (argCount ? args[argCount - 1]->endNode() : this);
    }

    exprNode* tupleNode::clone() const {
      return new tupleNode(token, args);
    }

    void tupleNode::setChildren(exprNodeRefVector &children) {
      const int argCount = (int) args.size();
      if (!argCount) {
        return;
      }

      children.reserve(argCount);
      for (int i = 0; i < argCount; ++i) {
        children.push_back(&(args[i]));
      }
    }

    void tupleNode::print(printer &pout) const {
      pout << '{';
      const int argCount = (int) args.size();
      for (int i = 0; i < argCount; ++i) {
        if (i) {
          pout << ", ";
        }
        pout << *(args[i]);
      }
      pout << '}';
    }

    void tupleNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---(tuple)\n";
      for (int i = 0; i < ((int) args.size()); ++i) {
        args[i]->childDebugPrint(prefix);
      }
    }
  }
}
