#include <occa/lang/expr/callNode.hpp>

namespace occa {
  namespace lang {
    callNode::callNode(token_t *token_,
                       const exprNode &value_,
                       const exprNodeVector &args_) :
      exprNode(token_),
      value(value_.clone()) {
      cloneExprNodeVector(args, args_);
    }

    callNode::callNode(const callNode &node) :
      exprNode(node.token),
      value(node.value->clone()) {
      cloneExprNodeVector(args, node.args);
    }

    callNode::~callNode() {
      delete value;
      freeExprNodeVector(args);
    }

    udim_t callNode::type() const {
      return exprNodeType::call;
    }

    exprNode* callNode::clone() const {
      return new callNode(token, *value, args);
    }

    exprNode* callNode::startNode() {
      return value->startNode();
    }

    exprNode* callNode::endNode() {
      const int argCount = (int) args.size();
      if (!argCount) {
        return value->endNode();
      }
      return args[argCount - 1]->endNode();
    }

    void callNode::setChildren(exprNodeRefVector &children) {
      const int argCount = (int) args.size();
      children.reserve(1 + argCount);

      children.push_back(&value);
      for (int i = 0; i < argCount; ++i) {
        children.push_back(&(args[i]));
      }
    }

    void callNode::print(printer &pout) const {
      pout << *value
           << '(';
      const int argCount = (int) args.size();
      for (int i = 0; i < argCount; ++i) {
        if (i) {
          pout << ", ";
        }
        pout << *(args[i]);
      }
      pout << ')';
    }

    void callNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << *value;
      io::stderr << "] (call)\n";
      for (int i = 0; i < ((int) args.size()); ++i) {
        args[i]->childDebugPrint(prefix);
      }
    }
  }
}
