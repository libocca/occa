#ifndef OCCA_LANG_EXPR_EXPRESSION_HEADER
#define OCCA_LANG_EXPR_EXPRESSION_HEADER

#include <list>
#include <vector>

#include <occa/lang/expr.hpp>

namespace occa {
  namespace lang {
    class expressionScopedState;

    typedef std::list<exprNode*>             exprNodeList;
    typedef std::list<token_t*>              tokenList;
    typedef std::list<exprOpNode*>           operatorList;
    typedef std::list<expressionScopedState> scopedStateList;

    //---[ Expression State ]-----------
    class expressionScopedState {
    public:
      token_t *beforePairToken;

      exprNodeList output;
      operatorList operators;

      expressionScopedState(token_t *beforePairToken_ = NULL);

      void free();

      void debugPrint();
    };

    class expressionState {
    public:
      tokenVector &tokens;

      // Keep track of the prev/next tokens
      //   to break ++ and -- left/right
      //   unary ambiguity
      token_t *prevToken;
      token_t *nextToken;

      // Token before the pair started
      token_t *beforePairToken;

      scopedStateList scopedStates;
      expressionScopedState *scopedState;

      exprNodeList usedOutput;
      operatorList usedOperators;

      bool hasError;

      expressionState(tokenVector &tokens_);
      ~expressionState();

      int outputCount();
      int operatorCount();

      exprNode& lastOutput();
      exprOpNode& lastOperator();

      void pushOutput(exprNode *expr);
      void pushOperator(operatorToken *token);
      void pushOperator(exprOpNode *expr);

      exprNode& unsafePopOutput();

      exprNode& popOutput();
      exprOpNode& popOperator();

      void pushPair(token_t *beforePairToken_);
      void popPair();

      void debugPrint();
    };
    //==================================

    // Using Shunting-Yard algorithm
    exprNode* getExpression(tokenVector &tokens);

    namespace expr {
      void getInitialExpression(tokenVector &tokens,
                                expressionState &state);

      void pushOutputNode(token_t *token,
                          expressionState &state);

      void closePair(expressionState &state);

      void extractArgs(exprNodeVector &args,
                       exprNode &node,
                       expressionState &state);

      void transformLastPair(operatorToken &opToken,
                             expressionState &state);

      void attachPair(operatorToken &opToken,
                      expressionState &state);

      bool operatorIsLeftUnary(operatorToken &opToken,
                               expressionState &state);

      void updateOperatorToken(operatorToken &opToken,
                               expressionState &state);

      void applyFasterOperators(operatorToken &opToken,
                                expressionState &state);

      void applyOperator(exprOpNode &opNode,
                         expressionState &state);

      void applyLeftUnaryOperator(exprOpNode &opNode,
                                  exprNode &value,
                                  expressionState &state);

      bool applyTernary(expressionState &state);
    }
  }
}

#endif
