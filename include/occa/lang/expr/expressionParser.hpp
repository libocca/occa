#ifndef OCCA_INTERNAL_LANG_EXPR_EXPRESSIONPARSER_HEADER
#define OCCA_INTERNAL_LANG_EXPR_EXPRESSIONPARSER_HEADER

#include <list>
#include <vector>

#include <occa/internal/lang/expr/exprNodes.hpp>

namespace occa {
  namespace lang {
    class expressionScopedState;

    typedef std::list<exprNode*>             exprNodeList;
    typedef std::list<token_t*>              tokenList;
    typedef std::list<exprOpNode*>           operatorList;
    typedef std::list<expressionScopedState> scopedStateList;

    //---[ Expression Scoped State ]----
    class expressionScopedState {
    public:
      token_t *beforePairToken;

      exprNodeList output;
      operatorList operators;

      expressionScopedState(token_t *beforePairToken_ = NULL);

      void free();

      void debugPrint();
    };
    //==================================


    //---[ Expression State ]-----------
    class expressionState {
    public:
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

      expressionState();
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


    //---[ Expression Parser ]----------
    class expressionParser {
      // Note: Uses the Shunting-Yard algorithm with customizations
      // to handle ambiguous operators
     private:
      tokenVector &tokens;
      expressionState state;

     public:
      expressionParser(tokenVector &tokens_);
      ~expressionParser();

      static exprNode* parse(tokenVector &tokens);

     private:
      exprNode* parse();

      void getInitialExpression();

      void pushOutputNode(token_t *token);

      void closePair();

      void extractArgs(exprNodeVector &args,
                       exprNode &node);

      void transformLastPair(operatorToken &opToken);

      void attachPair(operatorToken &opToken);

      bool operatorIsLeftUnary(operatorToken &opToken);

      void updateOperatorToken(operatorToken &opToken);

      void applyFasterOperators(operatorToken &opToken);

      void applyOperator(exprOpNode &opNode);

      void applyLeftUnaryOperator(exprOpNode &opNode,
                                  exprNode &value);

      bool applyTernary();
    };
    //==================================
  }
}

#endif
