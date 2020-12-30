#include <occa/internal/lang/expr/callNode.hpp>

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

    void callNode::pushChildNodes(exprNodeVector &children) {
      children.push_back(value);
      for (exprNode *arg : args) {
        children.push_back(arg);
      }
    }

    bool callNode::safeReplaceExprNode(exprNode *currentNode, exprNode *newNode) {
      if (currentNode == value) {
        delete value;
        value = newNode;
        return true;
      }

      for (exprNode *&arg : args) {
        if (currentNode == arg) {
          delete arg;
          arg = newNode;
          return true;
        }
      }

      return false;
    }

    void callNode::print(printer &pout) const {
      bool useNewlineDelimiters = false;
      std::string functionName = value->toString();
      int lineWidth = (
        pout.cursorPosition()
        + (int) functionName.size()
      );

      const int argCount = (int) args.size();
      for (int i = 0; i < argCount; ++i) {
        const std::string argStr = args[i]->toString();
        const int argSize = (int) argStr.size();
        lineWidth += argSize;

        useNewlineDelimiters |= (
          argSize > PRETTIER_MAX_VAR_WIDTH
          || lineWidth > PRETTIER_MAX_LINE_WIDTH
        );
      }

      pout << functionName
           << '(';

      if (useNewlineDelimiters) {
        pout.addIndentation();
        pout.printNewline();
        pout.printIndentation();
      }

      for (int i = 0; i < argCount; ++i) {
        if (i) {
          if (useNewlineDelimiters) {
            pout << ',';
            pout.printNewline();
            pout.printIndentation();
          } else {
            pout << ", ";
          }
        }
        pout << *(args[i]);
      }

      if (useNewlineDelimiters) {
        pout.removeIndentation();
        pout.printNewline();
        pout.printIndentation();
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
