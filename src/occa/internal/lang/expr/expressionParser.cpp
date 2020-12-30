#include <occa/internal/utils/string.hpp>
#include <occa/internal/lang/expr/expressionParser.hpp>

namespace occa {
  namespace lang {
    static const int outputTokenType = (tokenType::identifier |
                                        tokenType::type       |
                                        tokenType::vartype    |
                                        tokenType::variable   |
                                        tokenType::function   |
                                        tokenType::primitive  |
                                        tokenType::char_      |
                                        tokenType::string);

    //---[ Expression Scoped State ]----
    expressionScopedState::expressionScopedState(token_t *beforePairToken_) :
      beforePairToken(beforePairToken_) {}

    void expressionScopedState::free() {
      exprNodeList::iterator outputIt = output.begin();
      while (outputIt != output.end()) {
        delete *outputIt;
        ++outputIt;
      }
      output.clear();

      operatorList::iterator operatorIt = operators.begin();
      while (operatorIt != operators.end()) {
        delete *operatorIt;
        ++operatorIt;
      }
      operators.clear();
    }

    void expressionScopedState::debugPrint() {
      io::stdout << "Outputs:\n";
      exprNodeList::iterator it = output.begin();
      while (it != output.end()) {
        (*it)->debugPrint();
        ++it;
      }

      io::stdout << "Operators:\n";
      operatorList::iterator itOp = operators.begin();
      while (itOp != operators.end()) {
        io::stdout << '[' << *((*itOp)->token) << "]\n";
        ++itOp;
      }
    }
    //==================================

    //---[ Expression State ]-----------
    expressionState::expressionState() :
      prevToken(NULL),
      nextToken(NULL),
      beforePairToken(NULL),
      hasError(false) {
      scopedStates.push_back(expressionScopedState());
      scopedState = &(scopedStates.back());
    }

    expressionState::~expressionState() {
      while (scopedStates.size()) {
        scopedStates.back().free();
        scopedStates.pop_back();
      }
      while (usedOutput.size()) {
        delete usedOutput.back();
        usedOutput.pop_back();
      }
      while (usedOperators.size()) {
        delete usedOperators.back();
        usedOperators.pop_back();
      }
    }

    int expressionState::outputCount() {
      return (int) scopedState->output.size();
    }

    int expressionState::operatorCount() {
      return (int) scopedState->operators.size();
    }

    exprNode& expressionState::lastOutput() {
      return *(scopedState->output.back());
    }

    exprOpNode& expressionState::lastOperator() {
      return *(scopedState->operators.back());
    }

    void expressionState::pushOutput(exprNode *expr) {
      scopedState->output.push_back(expr);
    }

    void expressionState::pushOperator(operatorToken *token) {
      scopedState->operators.push_back(
        new exprOpNode(*token)
      );
    }

    void expressionState::pushOperator(exprOpNode *expr) {
      scopedState->operators.push_back(expr);
    }

    exprNode& expressionState::unsafePopOutput() {
      exprNode &ret = *(scopedState->output.back());
      scopedState->output.pop_back();
      return ret;
    }

    exprNode& expressionState::popOutput() {
      exprNode &ret = *(scopedState->output.back());
      usedOutput.push_back(&ret);
      scopedState->output.pop_back();
      return ret;
    }

    exprOpNode& expressionState::popOperator() {
      exprOpNode &ret = *(scopedState->operators.back());
      usedOperators.push_back(&ret);
      scopedState->operators.pop_back();
      return ret;
    }

    void expressionState::pushPair(token_t *beforePairToken_) {
      scopedStates.push_back(expressionScopedState(beforePairToken_));
      scopedState = &(scopedStates.back());

      beforePairToken = beforePairToken_;
    }

    void expressionState::popPair() {
      beforePairToken = scopedState->beforePairToken;

      expressionScopedState prevScopedState = scopedStates.back();
      scopedStates.pop_back();
      scopedState = &(scopedStates.back());

      // Copy left-overs
      scopedState->output.insert(scopedState->output.end(),
                                 prevScopedState.output.begin(),
                                 prevScopedState.output.end());

      scopedState->operators.insert(scopedState->operators.end(),
                                    prevScopedState.operators.begin(),
                                    prevScopedState.operators.end());
    }

    void expressionState::debugPrint() {
      io::stdout << "\n---[ Scopes ]---------------------------\n";
      scopedStateList::iterator it = scopedStates.begin();
      while (it != scopedStates.end()) {
        it->debugPrint();
        ++it;
        if (it != scopedStates.end()) {
          io::stdout << " - - - - - - - - - - - - - - - - - - - -\n";
        }
      }
      io::stdout << "========================================\n";
    }
    //==================================


    //---[ Expression Parser ]----------
    expressionParser::expressionParser(tokenVector &tokens_) :
        tokens(tokens_) {}

    expressionParser::~expressionParser() {
      freeTokenVector(tokens);
    }

    exprNode* expressionParser::parse(tokenVector &tokens_) {
      expressionParser parser(tokens_);
      return parser.parse();
    }

    exprNode* expressionParser::parse() {
      if (!tokens.size()) {
        return noExprNode.clone();
      }

      getInitialExpression();
      if (state.hasError) {
        return NULL;
      }

      // Finish applying operators
      while (state.operatorCount()) {
        applyOperator(state.popOperator());

        if (state.hasError) {
          return NULL;
        }
      }

      // Make sure we only have 1 root node
      const int outputCount = state.outputCount();
      if (!outputCount) {
        return noExprNode.clone();
      }

      applyTernary();

      if (outputCount > 1) {
        state.debugPrint();
        state.popOutput();
        state.lastOutput().token->printError("Unable to form an expression");
        return NULL;
      }

      return &(state.unsafePopOutput());
    }

    void expressionParser::getInitialExpression() {
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
          pushOutputNode(token);
        }
        else if (tokenType & tokenType::op) {
          operatorToken &opToken = token->to<operatorToken>();

          if (opToken.opType() & operatorType::pairStart) {
            state.pushPair(state.prevToken);
            state.pushOperator(&opToken);
          }
          else if (opToken.opType() & operatorType::pairEnd) {
            state.pushOperator(&opToken);
            state.popPair();
            closePair();
            if (!state.hasError) {
              attachPair(opToken);
            }
          }
          else {
            applyFasterOperators(opToken);
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

    void expressionParser::pushOutputNode(token_t *token) {
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
      else if (tokenType & tokenType::vartype) {
        vartypeToken &t = token->to<vartypeToken>();
        state.pushOutput(new vartypeNode(token, t.value));
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

    void expressionParser::closePair() {
      exprOpNode &opNode = state.popOperator();
      const operator_t &op = opNode.op;
      const opType_t opType = op.opType;
      operatorToken *errorToken = (operatorToken*) opNode.token;

      while (state.operatorCount()) {
        exprOpNode &nextOpNode = state.popOperator();
        const opType_t nextOpType = nextOpNode.opType();

        if (nextOpType & operatorType::pairStart) {
          if (opType == (nextOpType << 1)) {
            applyTernary();
            applyOperator(opNode);
            return;
          }
          errorToken = (operatorToken*) nextOpNode.token;
          break;
        }

        applyOperator(nextOpNode);

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

    void expressionParser::extractArgs(exprNodeVector &args,
                                       exprNode &node) {
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

    void expressionParser::transformLastPair(operatorToken &opToken) {
      // Guaranteed to have the pairNode
      pairNode &pair = state.popOutput().to<pairNode>();

      if (!(pair.opType() & (operatorType::parentheses |
                             operatorType::braces))) {
        state.hasError = true;
        state.debugPrint();
        opToken.printError("Expected identifier or proper expression before");
        return;
      }

      if (pair.opType() & operatorType::parentheses) {
        if (pair.value->type() & (exprNodeType::type |
                                  exprNodeType::vartype)) {
          state.pushOperator(
            new leftUnaryOpNode(&opToken,
                                op::parenCast,
                                *(pair.value))
          );
        } else {
          state.pushOutput(
            new parenthesesNode(pair.token,
                                *pair.value)
          );
        }
      } else {
        exprNodeVector args;
        extractArgs(args, *pair.value);
        state.pushOutput(
          new tupleNode(pair.token,
                        args)
        );
      }
    }

    void expressionParser::attachPair(operatorToken &opToken) {
      if ((state.outputCount() < 2)) {
        transformLastPair(opToken);
        return;
      }

      // Only consider () as a function call if preceeded by:
      //   - identifier
      //   - pairEnd
      const int prevTokenType = token_t::safeType(state.beforePairToken);
      if (!(prevTokenType & (outputTokenType |
                             tokenType::op))) {
        transformLastPair(opToken);
        return;
      }
      if (prevTokenType & tokenType::op) {
        operatorToken &prevOpToken = state.beforePairToken->to<operatorToken>();
        if (!(prevOpToken.opType() & operatorType::pairEnd)) {
          transformLastPair(opToken);
          return;
        }
      }

      pairNode &pair  = state.popOutput().to<pairNode>();
      exprNode &value = state.popOutput();

      // func(...)
      if (pair.opType() & operatorType::parentheses) {
        exprNodeVector args;
        extractArgs(args, *pair.value);
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
        extractArgs(args, *pair.value);

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

    bool expressionParser::operatorIsLeftUnary(operatorToken &opToken) {
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
      if (prevOpType & operatorType::pairStart) {
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

    void expressionParser::updateOperatorToken(operatorToken &opToken) {

      const opType_t opType = opToken.opType();
      if (!(opType & operatorType::ambiguous)) {
        return;
      }

      fileOrigin origin = opToken.origin;

      bool isLeftUnary = operatorIsLeftUnary(opToken);
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

    void expressionParser::applyFasterOperators(operatorToken &opToken) {

      updateOperatorToken(opToken);
      if (state.hasError) {
        return;
      }

      const operator_t &op = *(opToken.op);
      while (state.operatorCount()) {
        const operator_t &prevOp = state.lastOperator().op;

        if (prevOp.opType & operatorType::pairStart) {
          break;
        }

        if ((op.precedence > prevOp.precedence) ||
            ((op.precedence == prevOp.precedence) &&
             op::associativity[prevOp.precedence] == op::leftAssociative)) {

          applyOperator(state.popOperator());

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
    }

    void expressionParser::applyOperator(exprOpNode &opNode) {

      operatorToken &opToken = *((operatorToken*) opNode.token);
      const operator_t &op = opNode.op;
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
          applyLeftUnaryOperator(opNode, value);
          if (opType & operatorType::colon) {
            applyTernary();
          }
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

    void expressionParser::applyLeftUnaryOperator(exprOpNode &opNode,
                                                  exprNode &value) {

      operatorToken &opToken = *((operatorToken*) opNode.token);
      const unaryOperator_t &op = (unaryOperator_t&) opNode.op;
      const opType_t opType = op.opType;

      if (!(opType & operatorType::special)) {
        state.pushOutput(new leftUnaryOpNode(&opToken,
                                             (const unaryOperator_t&) op,
                                             value));
        return;
      }

      if (opType & operatorType::parenCast) {
        leftUnaryOpNode &parenOpNode = (leftUnaryOpNode&) opNode;
        exprNode *valueNode = parenOpNode.value;
        if (valueNode->type() & exprNodeType::type) {
          state.pushOutput(
            new parenCastNode(parenOpNode.token,
                              ((typeNode*) valueNode)->value,
                              value)
          );
        } else {
          state.pushOutput(
            new parenCastNode(parenOpNode.token,
                              ((vartypeNode*) valueNode)->value,
                              value)
          );
        }
      }
      else if (opType & operatorType::sizeof_) {
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

    bool expressionParser::applyTernary() {
      if (state.outputCount() < 3) {
        return false;
      }
      // Don't use state's garbage collection yet
      exprNode &falseValue = state.unsafePopOutput();
      exprNode &trueValue  = state.unsafePopOutput();
      exprNode &checkValue = state.unsafePopOutput();

      if ((trueValue.type() & exprNodeType::leftUnary)
          && (falseValue.type() & exprNodeType::leftUnary)) {

        leftUnaryOpNode &trueOpValue  = (leftUnaryOpNode&) trueValue;
        leftUnaryOpNode &falseOpValue = (leftUnaryOpNode&) falseValue;

        opType_t op1 = trueOpValue.opType();
        opType_t op2 = falseOpValue.opType();

        if ((op1 == operatorType::questionMark)
            && (op2 == operatorType::colon)) {

          state.pushOutput(
            new ternaryOpNode(checkValue,
                              *(trueOpValue.value),
                              *(falseOpValue.value))
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
