#ifndef OCCA_LANG_EXPR_EXPR_HEADER
#define OCCA_LANG_EXPR_EXPR_HEADER

namespace occa {
  namespace lang {
    class exprNode;
    class token_t;

    //---[ expr ]-----------------------
    class expr {
     public:
      exprNode *node;

      expr();
      expr(exprNode *node_);

      expr(const expr &other);
      ~expr();

      expr& operator = (exprNode *node_);
      expr& operator = (const expr &other);

      token_t* source() const;

      exprNode* popExprNode();

      expr operator [] (const expr &e);

      static expr parens(const expr &e);
    };
    //==================================

    //---[ Operators ]------------------
    expr operator + (const expr &left, const expr &right);
    expr operator * (const expr &left, const expr &right);
    //==================================
  }
}

#endif
