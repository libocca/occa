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

    expr expr::leftUnaryOpExpr(const unaryOperator_t &op_,
                               const expr &e) {
      return new leftUnaryOpNode(e.source(),
                                 op_,
                                 *e.node);
    }

    expr expr::rightUnaryOpExpr(const unaryOperator_t &op_,
                                const expr &e) {
      return new rightUnaryOpNode(e.source(),
                                  op_,
                                  *e.node);
    }

    expr expr::binaryOpExpr(const binaryOperator_t &op_,
                            const expr &left,
                            const expr &right) {
      return new binaryOpNode(left.source(),
                              op_,
                              *left.node,
                              *right.node);
    }

    // --e
    expr expr::operator -- () {
      return leftUnaryOpExpr(op::leftDecrement, *this);
    }

    // e--
    expr expr::operator -- (int) {
      return rightUnaryOpExpr(op::rightDecrement, *this);
    }

    // ++e
    expr expr::operator ++ () {
      return leftUnaryOpExpr(op::leftIncrement, *this);
    }

    // e++
    expr expr::operator ++ (int) {
      return rightUnaryOpExpr(op::rightIncrement, *this);
    }
    //==================================

    //---[ Operators ]------------------
    expr operator + (const expr &left, const expr &right) {
      return expr::binaryOpExpr(op::add, left, right);
    }

    expr operator - (const expr &left, const expr &right) {
      return expr::binaryOpExpr(op::sub, left, right);
    }

    expr operator * (const expr &left, const expr &right) {
      return expr::binaryOpExpr(op::mult, left, right);
    }

    expr operator / (const expr &left, const expr &right) {
      return expr::binaryOpExpr(op::div, left, right);
    }

    expr operator % (const expr &left, const expr &right) {
      return expr::binaryOpExpr(op::mod, left, right);
    }

    expr operator += (const expr &left, const expr &right) {
      return expr::binaryOpExpr(op::addEq, left, right);
    }

    expr operator -= (const expr &left, const expr &right) {
      return expr::binaryOpExpr(op::subEq, left, right);
    }

    expr operator *= (const expr &left, const expr &right) {
      return expr::binaryOpExpr(op::multEq, left, right);
    }

    expr operator /= (const expr &left, const expr &right) {
      return expr::binaryOpExpr(op::divEq, left, right);
    }

    expr operator < (const expr &left, const expr &right) {
      return expr::binaryOpExpr(op::lessThan, left, right);
    }

    expr operator <= (const expr &left, const expr &right) {
      return expr::binaryOpExpr(op::lessThanEq, left, right);
    }

    expr operator == (const expr &left, const expr &right) {
      return expr::binaryOpExpr(op::equal, left, right);
    }

    expr operator != (const expr &left, const expr &right) {
      return expr::binaryOpExpr(op::notEqual, left, right);
    }

    expr operator > (const expr &left, const expr &right) {
      return expr::binaryOpExpr(op::greaterThan, left, right);
    }

    expr operator >= (const expr &left, const expr &right) {
      return expr::binaryOpExpr(op::greaterThanEq, left, right);
    }
    //==================================
  }
}
