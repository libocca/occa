#ifndef OCCA_INTERNAL_LANG_EXPR_EXPR_HEADER
#define OCCA_INTERNAL_LANG_EXPR_EXPR_HEADER

#include <occa/internal/lang/operator.hpp>
#include <occa/types/primitive.hpp>

namespace occa {
  namespace lang {
    class exprNode;
    class token_t;
    class unaryOperator_t;
    class binaryOperator_t;
    class variable_t;
    class blockStatement;
    class expressionStatement;

    //---[ expr ]-----------------------
    class expr {
     public:
      exprNode *node;

      expr();
      expr(exprNode *node_);
      expr(exprNode &node_);

      expr(token_t *source_,
           const primitive &p);

      expr(variable_t &var);

      expr(token_t *source_, variable_t &var);

      expr(const expr &other);
      ~expr();

      expr& operator = (exprNode *node_);
      expr& operator = (const expr &other);

      static expr usingExprNode(exprNode *node_);

      token_t* source() const;
      const opType_t& opType() const;

      exprNode* cloneExprNode();

      exprNode* popExprNode();

      expr operator [] (const expr &e);

      expressionStatement* createStatement(blockStatement *up,
                                           const bool hasSemicolon = true);

      static expr parens(const expr &e);

      static expr leftUnaryOpExpr(const unaryOperator_t &op_,
                                  const expr &e);

      static expr rightUnaryOpExpr(const unaryOperator_t &op_,
                                   const expr &e);

      static expr binaryOpExpr(const binaryOperator_t &op_,
                               const expr &left,
                               const expr &right);

      // --e
      expr operator -- ();
      // e--
      expr operator -- (int);

      // ++e
      expr operator ++ ();
      // e++
      expr operator ++ (int);
    };
    //==================================

    //---[ Operators ]------------------
    expr operator + (const expr &left, const expr &right);
    expr operator - (const expr &left, const expr &right);
    expr operator * (const expr &left, const expr &right);
    expr operator / (const expr &left, const expr &right);
    expr operator % (const expr &left, const expr &right);

    expr operator += (const expr &left, const expr &right);
    expr operator -= (const expr &left, const expr &right);
    expr operator *= (const expr &left, const expr &right);
    expr operator /= (const expr &left, const expr &right);

    expr operator < (const expr &left, const expr &right);
    expr operator <= (const expr &left, const expr &right);
    expr operator == (const expr &left, const expr &right);
    expr operator != (const expr &left, const expr &right);
    expr operator > (const expr &left, const expr &right);
    expr operator >= (const expr &left, const expr &right);

    printer& operator << (printer &pout, const expr &e);
    //==================================
  }
}

#endif
