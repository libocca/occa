#include <occa/internal/lang/expr/tupleNode.hpp>

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

    void tupleNode::pushChildNodes(exprNodeVector &children) {
      for (exprNode *arg : args) {
        children.push_back(arg);
      }
    }

    bool tupleNode::safeReplaceExprNode(exprNode *currentNode, exprNode *newNode) {
      for (exprNode *arg : args) {
        if (currentNode == arg) {
          delete arg;
          arg = newNode;
          return true;
        }
      }

      return false;
    }

    void tupleNode::print(printer &pout) const {
      bool useNewlineDelimiters = false;
      strVector printedArgs;
      // For the initial {
      int lineWidth = pout.cursorPosition() + 1;

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

      pout << '{';

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
