#include <occa/internal/lang/expr/exprOpNode.hpp>
#include <occa/internal/lang/expr/parenthesesNode.hpp>

namespace occa {
  namespace lang {
    exprOpNode::exprOpNode(operatorToken &token_) :
      exprNode(&token_),
      op(*(token_.op)) {}

    exprOpNode::exprOpNode(token_t *token_,
                           const operator_t &op_) :
      exprNode(token_),
      op(op_) {}

    opType_t exprOpNode::opType() const {
      return op.opType;
    }

    udim_t exprOpNode::type() const {
      return exprNodeType::rawOp;
    }

    exprNode* exprOpNode::clone() const {
      return new exprOpNode(token, op);
    }

    exprNode* exprOpNode::wrapInParentheses() {
      return new parenthesesNode(token, *this);
    }

    void exprOpNode::print(printer &pout) const {
      token->printError("[Waldo] (exprOpNode) Unsure how you got here...");
    }

    void exprOpNode::debugPrint(const std::string &prefix) const {
      printer pout(io::stderr);
      io::stderr << prefix << "|\n"
                << prefix << "|---[";
      pout << op;
      io::stderr << "] (exprOpNode)\n";
    }
  }
}
