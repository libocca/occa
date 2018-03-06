/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */
#include "occa/tools/string.hpp"

#include "expression.hpp"

namespace occa {
  namespace lang {
    //---[ Expression State ]-----------
    expressionState::expressionState() :
      prevToken(NULL),
      nextToken(NULL),
      tokenBeforePair(NULL),
      hasError(false) {}

    expressionState::~expressionState() {
      // Tokens are freed outside in the case of an error
      if (!hasError) {
        while (operators.size()) {
          delete operators.top();
          operators.pop();
        }
      }
      while (output.size()) {
        delete output.top();
        output.pop();
      }
      while (usedOutput.size()) {
        delete usedOutput.top();
        usedOutput.pop();
      }
    }

    int expressionState::outputCount() {
      return (int) output.size();
    }

    int expressionState::operatorCount() {
      return (int) operators.size();
    }

    exprNode& expressionState::lastOutput() {
      return *(output.top());
    }

    operatorToken& expressionState::lastOperator() {
      return *(operators.top());
    }

    exprNode& expressionState::popOutput() {
      exprNode &ret = *(output.top());
      usedOutput.push(&ret);
      output.pop();
      return ret;
    }

    operatorToken& expressionState::popOperator() {
      operatorToken &ret = *(operators.top());
      operators.pop();
      return ret;
    }
    //==================================

    exprNode* getExpression(tokenVector &tokens) {
      if (!tokens.size()) {
        return &(noExprNode.clone());
      }

      // TODO: Ternary operator
      expressionState state;
      getInitialExpression(tokens, state);
      if (state.hasError) {
        freeTokenVector(tokens);
        return NULL;
      }

      // Finish applying operators
      while (state.operatorCount()) {
        applyOperator(state.popOperator(),
                      state);

        if (state.hasError) {
          freeTokenVector(tokens);
          return NULL;
        }
      }

      // Make sure we only have 1 root node
      const int outputCount = state.outputCount();
      if (!outputCount) {
        return &(noExprNode.clone());
      }
      if (outputCount > 1) {
        state.popOutput();
        state.lastOutput().token->printError("Unable to form an expression");
        freeTokenVector(tokens);
        return NULL;
      }

      freeTokenVector(tokens);

      // Pop output before state frees it
      // Do this manually to prevent freeing it
      exprNode *ret = state.output.top();
      state.output.pop();
      return ret;
    }

    void getInitialExpression(tokenVector &tokens,
                              expressionState &state) {

      const int outputTokenType = (tokenType::identifier |
                                   tokenType::primitive  |
                                   tokenType::char_      |
                                   tokenType::string);

      const int count = (int) tokens.size();
      for (int i = 0; i < count; ++i) {
        token_t *token = tokens[i];
        if (!token) {
          continue;
        }

        state.nextToken = NULL;
        for (int j = (i + 1); j < count; ++j) {
          if (tokens[j]) {
            state.nextToken = tokens[j];
            break;
          }
        }

        const int tokenType = token->type();
        if (tokenType & outputTokenType) {
          pushOutputNode(token, state);
        }
        else if (tokenType & tokenType::op) {
          operatorToken &opToken = token->to<operatorToken>();

          if (opToken.opType() & operatorType::pairStart) {
            state.tokenBeforePair = state.prevToken;
            state.operators.push(&opToken);
          }
          else if (opToken.opType() & operatorType::pairEnd) {
            closePair(opToken, state);
            if (!state.hasError) {
              attachPair(opToken, state);
            }
          }
          else {
            applyFasterOperators(opToken, state);
          }
        }

        if (state.hasError) {
          // ~state will delete seen tokens
          for (int j = 0; j <= i; ++j) {
            tokens[j] = NULL;
          }
          return;
        }
        // Store prevToken at the end since opToken
        //   might have changed from an ambiguous type
        state.prevToken = token;
      }
    }

    void pushOutputNode(token_t *token,
                        expressionState &state) {
      const int tokenType = token->type();
      if (tokenType & tokenType::identifier) {
        identifierToken &t = token->to<identifierToken>();
        state.output.push(new identifierNode(token, t.value));
      }
      else if (tokenType & tokenType::primitive) {
        primitiveToken &t = token->to<primitiveToken>();
        state.output.push(new primitiveNode(token, t.value));
      }
      else if (tokenType & tokenType::char_) {
        // TODO: Handle char udfs here
        charToken &t = token->to<charToken>();
        state.output.push(new charNode(token, t.value));
      }
      else if (tokenType & tokenType::string) {
        // TODO: Handle string udfs here
        stringToken &t = token->to<stringToken>();
        state.output.push(new stringNode(token, t.value));
      }
    }

    void closePair(operatorToken &opToken,
                   expressionState &state) {
      const opType_t opType = opToken.opType();
      operatorToken *errorToken = &opToken;

      while (state.operatorCount()) {
        operatorToken &nextOpToken = state.popOperator();
        const opType_t nextOpType = nextOpToken.opType();

        if (nextOpType & operatorType::pairStart) {
          if (opType == (nextOpType << 1)) {
            applyOperator(opToken, state);
            return;
          }
          errorToken = &nextOpToken;
          break;
        }

        applyOperator(nextOpToken, state);

        if (state.hasError) {
          return;
        }
      }

      // Found a pairStart that doesn't match
      state.hasError = true;

      std::stringstream ss;
      ss << "Could not find a closing '"
         << ((pairOperator_t*) errorToken->op)->pairStr
         << '\'';
      errorToken->printError(ss.str());
    }

    void extractArgs(exprNodeVector &args,
                     exprNode &node,
                     expressionState &state) {
      // We need to push all args and reverse it at the end
      //   since commaNode looks like (...tail, head)
      exprNode *commaNode = &node;
      while (true) {
        if (commaNode->type() & exprNodeType::binary) {
          binaryOpNode &opNode = commaNode->to<binaryOpNode>();
          if (opNode.opType() & operatorType::comma) {
            args.push_back(&opNode.rightValue);
            commaNode = &(opNode.leftValue);
            continue;
          }
        }
        args.push_back(commaNode);
        break;
      }

      // Reverse arguments back to original order
      const int argCount = (int) args.size();
      for (int i = 0 ; i < (argCount / 2); ++i) {
        exprNode *arg_i = args[i];
        args[i] = args[argCount - i - 1];
        args[argCount - i - 1] = arg_i;
      }
    }

    void transformLastPair(operatorToken &opToken,
                           expressionState &state) {
      // Guaranteed to have the pairNode
      pairNode &pair = state.popOutput().to<pairNode>();

      if (!(pair.opType() & (operatorType::parentheses |
                              operatorType::braces))) {
        state.hasError = true;
        opToken.printError("Expected identifier or proper expression before");
        return;
      }

      if (pair.opType() & operatorType::parentheses) {
        state.output.push(new parenthesesNode(pair.token,
                                              pair.value));
      } else {
        exprNodeVector args;
        extractArgs(args, pair.value, state);
        state.output.push(new tupleNode(pair.token,
                                        args));
      }
    }

    void attachPair(operatorToken &opToken,
                    expressionState &state) {
      if (state.outputCount() < 2) {
        transformLastPair(opToken, state);
        return;
      }

      // Only consider () as a function call if preceeded by:
      //   - identifier
      //   - pairEnd
      const int prevTokenType = state.tokenBeforePair->type();
      if (!(prevTokenType & (tokenType::identifier |
                             tokenType::op))) {
        transformLastPair(opToken, state);
        return;
      }
      if (prevTokenType & tokenType::op) {
        operatorToken &prevOpToken = state.tokenBeforePair->to<operatorToken>();
        if (!(prevOpToken.opType() & operatorType::pairEnd)) {
          transformLastPair(opToken, state);
          return;
        }
      }

      pairNode &pair  = state.popOutput().to<pairNode>();
      exprNode &value = state.popOutput();

      // func(...)
      if (pair.opType() & operatorType::parentheses) {
        exprNodeVector args;
        extractArgs(args, pair.value, state);
        state.output.push(new callNode(value.token,
                                       value,
                                       args));
        return;
      }
      // array[...]
      if (pair.opType() & operatorType::brackets) {
        state.output.push(new subscriptNode(value.token,
                                            value,
                                            pair.value));
        return;
      }
      // func<<<...>>>
      if (pair.opType() & operatorType::cudaCall) {
        exprNodeVector args;
        extractArgs(args, pair.value, state);

        const int argCount = (int) args.size();
        if (argCount == 1) {
          args[0]->token->printError("Must also have threads per block"
                                     " as the second argument");
          state.hasError = true;
        } else if (argCount > 2) {
          args[0]->token->printError("Kernel call only takes 2 arguments");
          state.hasError = true;
        }

        if (!state.hasError) {
          state.output.push(new cudaCallNode(value.token,
                                             value,
                                             *args[0],
                                             *args[1]));
        }
        return;
      }

      state.hasError = true;
      opToken.printError("[Waldo] (attachPair) Unsure how you got here...");
    }

    bool operatorIsLeftUnary(operatorToken &opToken,
                             expressionState &state) {
      const opType_t opType = opToken.opType();

      // Test for chaining increments
      // 1 + ++ ++ x
      // (2) ++ ++
      opType_t chainable = (operatorType::increment |
                            operatorType::decrement |
                            operatorType::parentheses);

      // ++ and -- operators
      const bool onlyUnary = (opType & (operatorType::increment |
                                        operatorType::decrement));

      // If this is the first token, it's left unary
      // If this is the last token, it's binary or right unary
      if ((!state.prevToken) != (!state.nextToken)) {
        return !state.prevToken;
      }

      // Test for left unary first
      const bool prevTokenIsOp = (state.prevToken->type() & tokenType::op);
      if (prevTokenIsOp) {
        opType_t prevType = state.prevToken->to<operatorToken>().opType();
        // + + + 1
        if (prevType & operatorType::leftUnary) {
          return true;
        }
        if (!onlyUnary) {
          return false;
        }
      }

      const bool nextTokenIsOp = (state.nextToken->type() & tokenType::op);

      //   v check right
      // 1 + ++ x
      //     ^ check left
      if (prevTokenIsOp != nextTokenIsOp) {
        return (onlyUnary
                ? prevTokenIsOp
                : nextTokenIsOp);
      }
      // y ++ x (Unable to apply operator)
      // y + x
      if (!prevTokenIsOp) {
        if (onlyUnary) {
          state.hasError = true;
          opToken.printError("Ambiguous operator");
        }
        return false;
      }

      opType_t prevType = state.prevToken->to<operatorToken>().opType();
      opType_t nextType = state.nextToken->to<operatorToken>().opType();

      // x ++ ++ ++ y
      if ((prevType & chainable) &&
          (nextType & chainable)) {
        state.hasError = true;
        opToken.printError("Ambiguous operator");
        return false;
      }

      return !(prevType & chainable);
    }

    void updateOperatorToken(operatorToken &opToken,
                             expressionState &state) {

      const opType_t opType = opToken.opType();
      if (!(opType & operatorType::ambiguous)) {
        return;
      }

      fileOrigin origin = opToken.origin;

      const bool isLeftUnary = operatorIsLeftUnary(opToken, state);
      if (state.hasError) {
        return;
      }

      const operator_t *newOperator = NULL;
      if (opType & operatorType::plus) {           // +
        newOperator = (isLeftUnary
                       ? (const operator_t*) &op::positive
                       : (const operator_t*) &op::add);
      }
      else if (opType & operatorType::minus) {     // -
        newOperator = (isLeftUnary
                       ? (const operator_t*) &op::negative
                       : (const operator_t*) &op::sub);
      }
      else if (opType & operatorType::asterisk) {  // *
        newOperator = (isLeftUnary
                       ? (const operator_t*) &op::dereference
                       : (const operator_t*) &op::mult);
      }
      else if (opType & operatorType::ampersand) { // &
        newOperator = (isLeftUnary
                       ? (const operator_t*) &op::address
                       : (const operator_t*) &op::bitAnd);
      }
      else if (opType & operatorType::increment) { // ++
        newOperator = (isLeftUnary
                       ? (const operator_t*) &op::leftIncrement
                       : (const operator_t*) &op::rightIncrement);
      }
      else if (opType & operatorType::decrement) { // --
        newOperator = (isLeftUnary
                       ? (const operator_t*) &op::leftDecrement
                       : (const operator_t*) &op::rightDecrement);
      }
      else if (opType & operatorType::scope) {     // ::
        newOperator = (isLeftUnary
                       ? (const operator_t*) &op::globalScope
                       : (const operator_t*) &op::scope);
      }

      if (newOperator) {
        opToken.op = newOperator;
        return;
      }

      state.hasError = true;
      opToken.printError("Unable to parse ambiguous token");
    }

    void applyFasterOperators(operatorToken &opToken,
                              expressionState &state) {

      updateOperatorToken(opToken, state);
      if (state.hasError) {
        return;
      }

      const operator_t &op = *(opToken.op);
      while (state.operatorCount()) {
        const operator_t &prevOp = *(state.lastOperator().op);

        if (prevOp.opType & operatorType::pairStart) {
          break;
        }

        if ((op.precedence > prevOp.precedence) ||
            ((op.precedence == prevOp.precedence) &&
             op::associativity[prevOp.precedence] == op::leftAssociative)) {

          applyOperator(state.popOperator(),
                        state);

          if (state.hasError) {
            return;
          }
          continue;
        }

        break;
      }

      // After applying faster operators,
      //   place opToken in the stack
      state.operators.push(&opToken);
    }

    void applyOperator(operatorToken &opToken,
                       expressionState &state) {

      const operator_t &op = *(opToken.op);
      const opType_t opType = op.opType;
      const int outputCount = state.outputCount();

      if (!outputCount) {
        state.hasError = true;
        opToken.printError("Unable to apply operator");
        return;
      }

      exprNode &value = state.popOutput();
      if (opType & operatorType::binary) {
        if (!outputCount) {
          state.hasError = true;
          opToken.printError("Unable to apply operator");
          return;
        }
        exprNode &left = state.popOutput();
        state.output.push(new binaryOpNode(&opToken,
                                           (const binaryOperator_t&) op,
                                           left,
                                           value));
      }
      else if (opType & operatorType::leftUnary) {
        applyLeftUnaryOperator(opToken,
                               (const unaryOperator_t&) op,
                               value,
                               state);
      }
      else if (opType & operatorType::rightUnary) {
        state.output.push(new rightUnaryOpNode(&opToken,
                                               (const unaryOperator_t&) op,
                                               value));
      }
      else if (opType & operatorType::pair) {
        state.output.push(new pairNode(opToken,
                                       value));
      } else {
        state.hasError = true;
        opToken.printError("Unable to apply operator");
      }
    }

    void applyLeftUnaryOperator(operatorToken &opToken,
                                const unaryOperator_t &op,
                                exprNode &value,
                                expressionState &state) {

      const opType_t opType = op.opType;
      if (!(opType & operatorType::special)) {
        state.output.push(new leftUnaryOpNode(&opToken,
                                              (const unaryOperator_t&) op,
                                              value));
        return;
      }

      // Handle new and delete
      if (opType & operatorType::sizeof_) {
        state.output.push(new sizeofNode(&opToken,
                                         value));
      }
      else if (opType & operatorType::new_) {
        state.hasError = true;
        opToken.printError("'new' not supported yet");
      }
      else if (opType & operatorType::delete_) {
        state.hasError = true;
        opToken.printError("'delete' not supported yet");
      }
      else if (opType & operatorType::throw_) {
        state.output.push(new throwNode(&opToken,
                                        value));
      }
      else {
        state.hasError = true;
        opToken.printError("[Waldo] (applyLeftUnaryOperator)"
                           " Unsure how you got here...");
      }
    }
  }
}
