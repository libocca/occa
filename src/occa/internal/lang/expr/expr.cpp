#include <occa/internal/utils/lex.hpp>
#include <occa/internal/lang/expr/expr.hpp>
#include <occa/internal/lang/expr/exprNodes.hpp>
#include <occa/internal/lang/statement.hpp>
#include <occa/internal/lang/variable.hpp>

namespace occa {
  namespace lang {
    //---[ expr ]-----------------------
    expr::expr() :
        node(NULL) {}

    expr::expr(exprNode *node_) :
        node(exprNode::clone(node_)) {}

    expr::expr(exprNode &node_) :
        node(node_.clone()) {}

    expr::expr(token_t *source_,
               const primitive &p) :
        node(new primitiveNode(source_, p)) {}

    expr::expr(variable_t &var) :
        node(new variableNode(var.source, var)) {}

    expr::expr(token_t *source_, variable_t &var) :
        node(new variableNode(source_, var)) {}

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

    expr expr::usingExprNode(exprNode *node_) {
      expr e;
      e.node = node_;
      return e;
    }

    token_t* expr::source() const {
      return node ? node->token : NULL;
    }

    const opType_t& expr::opType() const {
      exprOpNode *opNode = dynamic_cast<exprOpNode*>(node);
      if (opNode) {
        return opNode->op.opType;
      }
      return operatorType::none;
    }

    exprNode* expr::cloneExprNode() {
      return exprNode::clone(node);
    }

    exprNode* expr::popExprNode() {
      exprNode *n = node;
      node = NULL;
      return n;
    }

    expr expr::operator [] (const expr &e) {
      return usingExprNode(
        new subscriptNode(source(),
                          *node,
                          *e.node)
      );
    }

    expressionStatement* expr::createStatement(blockStatement *up,
                                               const bool hasSemicolon) {
      return new expressionStatement(up, *node->clone(), hasSemicolon);
    }

    expr expr::parens(const expr &e) {
      if (!e.node) {
        return expr();
      }
      return usingExprNode(
        e.node->wrapInParentheses()
      );
    }

    expr expr::leftUnaryOpExpr(const unaryOperator_t &op_,
                               const expr &e) {
      return usingExprNode(
        new leftUnaryOpNode(e.source(),
                            op_,
                            *e.node)
      );
    }

    expr expr::rightUnaryOpExpr(const unaryOperator_t &op_,
                                const expr &e) {
      return usingExprNode(
        new rightUnaryOpNode(e.source(),
                             op_,
                             *e.node)
      );
    }

    expr expr::binaryOpExpr(const binaryOperator_t &op_,
                            const expr &left,
                            const expr &right) {
      return usingExprNode(
        new binaryOpNode(left.source(),
                         op_,
                         *left.node,
                         *right.node)
      );
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

    printer& operator << (printer &pout, const expr &e) {
      if (e.node) {
        e.node->print(pout);
      }
      return pout;
    }
    //==================================
  }
}
