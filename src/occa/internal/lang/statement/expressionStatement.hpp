#ifndef OCCA_INTERNAL_LANG_STATEMENT_EXPRESSIONSTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_EXPRESSIONSTATEMENT_HEADER

#include <occa/internal/lang/statement/statement.hpp>

namespace occa {
  namespace lang {
    class expressionStatement : public statement_t {
    public:
      exprNode *expr;
      bool hasSemicolon;

      expressionStatement(blockStatement *up_,
                          exprNode &expr_,
                          const bool hasSemicolon_ = true);
      expressionStatement(blockStatement *up_,
                          const expressionStatement &other);
      ~expressionStatement();

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      virtual exprNodeArray getDirectExprNodes();

      virtual void safeReplaceExprNode(exprNode *currentNode, exprNode *newNode);

      virtual void print(printer &pout) const;
    };
  }
}

#endif
