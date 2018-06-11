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
#include <occa/tools/string.hpp>

#include <occa/lang/expression.hpp>

namespace occa {
  namespace lang {
    static const int outputTokenType = (tokenType::identifier |
                                        tokenType::type       |
                                        tokenType::variable   |
                                        tokenType::function   |
                                        tokenType::primitive  |
                                        tokenType::char_      |
                                        tokenType::string);

    //---[ Expression State ]-----------
    expressionState::expressionState(tokenVector &tokens_) :
      tokens(tokens_),
      prevToken(NULL),
      nextToken(NULL),
      hasError(false) {}

    expressionState::~expressionState() {
      freeTokenVector(tokens);
      while (output.size()) {
        delete output.top();
        output.pop();
      }
      while (usedOutput.size()) {
        delete usedOutput.top();
        usedOutput.pop();
      }
    }

    token_t* expressionState::tokenBeforePair() {
      if (!tokensBeforePair.size()) {
        return NULL;
      }
      return tokensBeforePair.top();
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

    void expressionState::pushOutput(exprNode *expr) {
      output.push(expr);
    }

    void expressionState::pushOperator(operatorToken *token) {
      operators.push(token);
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
        return noExprNode.clone();
      }

      // TODO: Ternary operator
      expressionState state(tokens);
      getInitialExpression(tokens, state);
      if (state.hasError) {
        return NULL;
      }

      // Finish applying operators
      while (state.operatorCount()) {
        if (applyTernary(state)) {
          continue;
        }

        applyOperator(state.popOperator(),
                      state);

        if (state.hasError) {
          return NULL;
        }
      }

      // Make sure we only have 1 root node
      const int outputCount = state.outputCount();
      if (!outputCount) {
        return noExprNode.clone();
      }

      applyTernary(state);

      if (outputCount > 1) {
        state.popOutput();
        state.lastOutput().token->printError("Unable to form an expression");
        return NULL;
      }

      // Pop output before state frees it
      // Do this manually to prevent freeing it
      exprNode *ret = state.output.top();
      state.output.pop();
      return ret;
    }

    void getInitialExpression(tokenVector &tokens,
                              expressionState &state) {
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
            state.tokensBeforePair.push(state.prevToken);
            state.pushOperator(&opToken);
          }
          else if (opToken.opType() & operatorType::pairEnd) {
            closePair(opToken, state);
            if (!state.hasError) {
              attachPair(opToken, state);
            }
            state.tokensBeforePair.pop();
          }
          else {
            applyFasterOperators(opToken, state);
          }
        }

        if (state.hasError) {
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
        state.pushOutput(new identifierNode(token, t.value));
      }
      else if (tokenType & tokenType::variable) {
        variableToken &t = token->to<variableToken>();
        state.pushOutput(new variableNode(token, t.value));
      }
      else if (tokenType & tokenType::function) {
        functionToken &t = token->to<functionToken>();
        state.pushOutput(new functionNode(token, t.value));
      }
      else if (tokenType & tokenType::type) {
        typeToken &t = token->to<typeToken>();
        state.pushOutput(new typeNode(token, t.value));
      }
      else if (tokenType & tokenType::primitive) {
        primitiveToken &t = token->to<primitiveToken>();
        state.pushOutput(new primitiveNode(token, t.value));
      }
      else if (tokenType & tokenType::char_) {
        // TODO: Handle char udfs here
        charToken &t = token->to<charToken>();
        state.pushOutput(new charNode(token, t.value));
      }
      else if (tokenType & tokenType::string) {
        // TODO: Handle string udfs here
        stringToken &t = token->to<stringToken>();
        state.pushOutput(new stringNode(token, t.value));
      }
    }

    void closePair(operatorToken &opToken,
                   expressionState &state) {
      const opType_t opType = opToken.opType();
      operatorToken *errorToken = &opToken;

      while (state.operatorCount()) {
        if (applyTernary(state)) {
          continue;
        }

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
      ss << "Could not find ";
      if (errorToken->opType() & operatorType::pairStart) {
        ss << "a closing";
      } else {
        ss << "an opening";
      }
      ss << " '"
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
            args.push_back(opNode.rightValue);
            commaNode = opNode.leftValue;
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
        state.pushOutput(
          new parenthesesNode(pair.token,
                              *pair.value)
        );
      } else {
        exprNodeVector args;
        extractArgs(args, *pair.value, state);
        state.pushOutput(
          new tupleNode(pair.token,
                        args)
        );
      }
    }

    void attachPair(operatorToken &opToken,
                    expressionState &state) {
      if ((state.outputCount() < 2)) {
        transformLastPair(opToken, state);
        return;
      }

      // Only consider () as a function call if preceeded by:
      //   - identifier
      //   - pairEnd
      const int prevTokenType = state.tokenBeforePair()->type();
      if (!(prevTokenType & (outputTokenType |
                             tokenType::op))) {
        transformLastPair(opToken, state);
        return;
      }
      if (prevTokenType & tokenType::op) {
        operatorToken &prevOpToken = state.tokenBeforePair()->to<operatorToken>();
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
        extractArgs(args, *pair.value, state);
        state.pushOutput(
          new callNode(value.token,
                       value,
                       args)
        );
        return;
      }
      // array[...]
      if (pair.opType() & operatorType::brackets) {
        state.pushOutput(
          new subscriptNode(value.token,
                            value,
                            *pair.value)
        );
        return;
      }
      // func<<<...>>>
      if (pair.opType() & operatorType::cudaCall) {
        exprNodeVector args;
        extractArgs(args, *pair.value, state);

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
          state.pushOutput(
            new cudaCallNode(value.token,
                             value,
                             *args[0],
                             *args[1])
          );
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
      if (!state.prevToken ||
          !state.nextToken) {
        return !state.prevToken;
      }

      opType_t prevOpType = state.prevToken->getOpType();
      if (prevOpType & (operatorType::pairStart |
                        operatorType::colon     |
                        operatorType::questionMark)) {
        return true;
      }

      // Test for left unary first
      const bool prevTokenIsOp = prevOpType & (operatorType::unary |
                                             operatorType::binary);
      if (prevTokenIsOp) {
        // + + + 1
        // a = + 1
        if ((prevOpType & operatorType::leftUnary) ||
            ((prevOpType & operatorType::binary) &&
             !(prevOpType & operatorType::unary))) {
          return true;
        }
        if (!onlyUnary) {
          return false;
        }
      }

      const bool nextTokenIsOp = (
        state.nextToken->getOpType() & (operatorType::unary |
                                        operatorType::binary)
      );

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

      opType_t nextOpType = state.nextToken->to<operatorToken>().opType();

      // x ++ ++ ++ y
      if ((prevOpType & chainable) &&
          (nextOpType & chainable)) {
        state.hasError = true;
        opToken.printError("Ambiguous operator");
        return false;
      }

      return !(prevOpType & chainable);
    }

    void updateOperatorToken(operatorToken &opToken,
                             expressionState &state) {

      const opType_t opType = opToken.opType();
      if (!(opType & operatorType::ambiguous)) {
        return;
      }

      fileOrigin origin = opToken.origin;

      bool isLeftUnary = true;
      bool stillAmbiguous = true;

      // Test if in ternary
      if (state.outputCount()
          && (state.lastOutput().type() & exprNodeType::rightUnary)) {

        rightUnaryOpNode &lastOutput = (rightUnaryOpNode&) state.lastOutput();
        if (lastOutput.opType() & (operatorType::questionMark |
                                   operatorType::colon)) {
          stillAmbiguous = false;
        }
      }

      if (stillAmbiguous) {
        isLeftUnary = operatorIsLeftUnary(opToken, state);
        if (state.hasError) {
          return;
        }
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
      state.pushOperator(&opToken);

      // Apply ternary operators immediately
      if (op.opType & (operatorType::questionMark |
                       operatorType::colon)) {
        applyOperator(state.popOperator(),
                      state);
      }
    }

    void applyOperator(operatorToken &opToken,
                       expressionState &state) {

      const operator_t &op = *(opToken.op);
      const opType_t opType = op.opType;
      const int outputCount = state.outputCount();

      if (opType & operatorType::binary) {
        if (outputCount >= 2) {
          exprNode &right = state.popOutput();
          exprNode &left = state.popOutput();
          state.pushOutput(
            new binaryOpNode(&opToken,
                             (const binaryOperator_t&) op,
                             left,
                             right)
          );
          return;
        }
        state.hasError = true;
      }
      else if (opType & operatorType::leftUnary) {
        if (outputCount >= 1) {
          exprNode &value = state.popOutput();
          applyLeftUnaryOperator(opToken,
                                 (const unaryOperator_t&) op,
                                 value,
                                 state);
          return;
        }
        state.hasError = true;
      }
      else if (opType & operatorType::rightUnary) {
        if (outputCount >= 1) {
          exprNode &value = state.popOutput();
          state.pushOutput(
            new rightUnaryOpNode(&opToken,
                                 (const unaryOperator_t&) op,
                                 value)
          );
          return;
        }
        state.hasError = true;
      }
      else if (opType & operatorType::pair) {
        // Make sure we have content in the parentheses
        if ((outputCount >= 1)
            && !(state.prevToken->getOpType() & operatorType::pairStart)) {
          exprNode &value = state.popOutput();
          state.pushOutput(
            new pairNode(opToken, value)
          );
        } else {
          state.pushOutput(
            new pairNode(opToken, noExprNode)
          );
        }
      }
      if (state.hasError) {
        opToken.printError("Unable to apply operator");
      }
    }

    void applyLeftUnaryOperator(operatorToken &opToken,
                                const unaryOperator_t &op,
                                exprNode &value,
                                expressionState &state) {

      const opType_t opType = op.opType;
      if (!(opType & operatorType::special)) {
        state.pushOutput(new leftUnaryOpNode(&opToken,
                                             (const unaryOperator_t&) op,
                                             value));
        return;
      }

      // Handle new and delete
      if (opType & operatorType::sizeof_) {
        state.pushOutput(
          new sizeofNode(&opToken, value)
        );
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
        state.pushOutput(
          new throwNode(&opToken, value)
        );
      }
      else {
        state.hasError = true;
        opToken.printError("[Waldo] (applyLeftUnaryOperator)"
                           " Unsure how you got here...");
      }
    }

    bool applyTernary(expressionState &state) {
      if (state.outputCount() < 3) {
        return false;
      }
      exprNode &falseValue = state.popOutput();
      exprNode &trueValue  = state.popOutput();
      exprNode &checkValue = state.popOutput();
      // Don't use state's garbage collection yet
      state.usedOutput.pop();
      state.usedOutput.pop();
      state.usedOutput.pop();

      if ((checkValue.type() & exprNodeType::rightUnary)
          && (trueValue.type() & exprNodeType::rightUnary)) {

        rightUnaryOpNode &checkOpValue = (rightUnaryOpNode&) checkValue;
        rightUnaryOpNode &trueOpValue  = (rightUnaryOpNode&) trueValue;

        opType_t op1 = checkOpValue.opType();
        opType_t op2 = trueOpValue.opType();

        if ((op1 == operatorType::questionMark)
            && (op2 == operatorType::colon)) {

          state.pushOutput(
            new ternaryOpNode(*(checkOpValue.value),
                              *(trueOpValue.value),
                              falseValue)
          );
          // Manually delete since we're avoiding garbage collection
          delete &checkValue;
          delete &trueValue;
          delete &falseValue;

          return true;
        }
      }

      state.pushOutput(&checkValue);
      state.pushOutput(&trueValue);
      state.pushOutput(&falseValue);
      return false;
    }
  }
}
