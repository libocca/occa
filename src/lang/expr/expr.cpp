#include <occa/lang/expr/expr.hpp>
#include <occa/lang/expr/exprNodes.hpp>

namespace occa {
  namespace lang {
    //---[ expr ]-----------------------
    expr::expr() :
        node(NULL) {}

    expr::expr(exprNode *node_) :
        node(exprNode::clone(node_)) {}

    expr::expr(const expr &other) :
        node(exprNode::clone(other.node)) {}

    expr::~expr() {
      delete node;
    }

    expr& expr::operator = (exprNode *node_) {
      delete node;
      node = exprNode::clone(node_);
      return *this;
    }

    expr& expr::operator = (const expr &other) {
      delete node;
      node = exprNode::clone(other.node);
      return *this;
    }


    token_t* expr::source() const {
      return node ? node->token : NULL;
    }

    exprNode* expr::popExprNode() {
      exprNode *n = node;
      node = NULL;
      return n;
    }

    expr expr::operator [] (const expr &e) {
      return new subscriptNode(source(),
                               *node,
                               *e.node);
    }

    expr expr::parens(const expr &e) {
      if (!e.node) {
        return expr();
      }
      return e.node->wrapInParentheses();
    }
    //==================================

    //---[ Operators ]------------------
    expr operator + (const expr &left, const expr &right) {
      return new binaryOpNode(left.source(),
                              op::add,
                              *left.node,
                              *right.node);
    }

    expr operator * (const expr &left, const expr &right) {
      return new binaryOpNode(left.source(),
                              op::mult,
                              *left.node,
                              *right.node);
    }
    //==================================
  }
}
