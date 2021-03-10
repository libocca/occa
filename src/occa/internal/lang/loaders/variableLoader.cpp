#include <occa/internal/lang/loaders/attributeLoader.hpp>
#include <occa/internal/lang/loaders/typeLoader.hpp>
#include <occa/internal/lang/loaders/variableLoader.hpp>
#include <occa/internal/lang/parser.hpp>
#include <occa/internal/lang/statementContext.hpp>
#include <occa/internal/lang/token.hpp>
#include <occa/internal/lang/tokenContext.hpp>
#include <occa/internal/lang/variable.hpp>

namespace occa {
  namespace lang {
    variableLoader_t::variableLoader_t(tokenContext_t &tokenContext_,
                                       statementContext_t &smntContext_,
                                       parser_t &parser_,
                                       nameToAttributeMap &attributeMap_) :
      tokenContext(tokenContext_),
      smntContext(smntContext_),
      parser(parser_),
      attributeMap(attributeMap_),
      success(true) {}

    bool variableLoader_t::loadVariable(variable_t &var) {
      vartype_t vartype;
      return loadVariable(vartype, var);
    }

    bool variableLoader_t::loadVariable(vartype_t &vartype,
                                        variable_t &var) {
      attributeTokenMap attrs;
      success = loadAttributes(tokenContext,
                               smntContext,
                               parser,
                               attributeMap,
                               attrs);
      if (!success) {
        return false;
      }

      success = loadType(tokenContext, smntContext, parser, vartype);
      if (!success) {
        return false;
      }

      success = (
        !isLoadingFunctionPointer()
        ? loadBasicVariable(vartype, var)
        : loadFunctionPointer(vartype, var)
      );
      if (!success) {
        return false;
      }

      success = loadAttributes(tokenContext,
                               smntContext,
                               parser,
                               attributeMap,
                               attrs);
      if (!success) {
        return false;
      }

      if (var.vartype.type) {
        var.attributes = var.vartype.type->attributes;
      }
      var.attributes.insert(attrs.begin(), attrs.end());

      return true;
    }

    bool variableLoader_t::isLoadingVariable() {
      const int tokens = tokenContext.size();
      // Variable must have an identifier token next
      if (!tokens ||
          (!(tokenContext[0]->type() & tokenType::identifier))) {
        return false;
      }
      // If nothing else follows, it must be a variable
      if (tokens == 1) {
        return true;
      }
      // Last check is if the variable is being called
      //   with a constructor vs defining a function:
      //   - int foo((...));
      // Note: We're guaranteed an extra token since we check for
      //         closing pairs. So there is at least one ')' token
      return (
        !(token_t::safeOperatorType(tokenContext[1]) & operatorType::parenthesesStart) ||
        (token_t::safeOperatorType(tokenContext[2]) & operatorType::parenthesesStart)
      );
    }

    bool variableLoader_t::isLoadingFunction() {
      tokenContext.push();

      tokenContext.supressErrors = true;
      vartype_t vartype;
      success = loadType(tokenContext, smntContext, parser, vartype);
      tokenContext.supressErrors = false;

      if (!success) {
        success = true;
        tokenContext.pop();
        return false;
      }

      if (!(token_t::safeType(tokenContext[0]) & tokenType::identifier)) {
        tokenContext.pop();
        return false;
      }

      const bool isFunction = (
        token_t::safeOperatorType(tokenContext[1]) & operatorType::parenthesesStart
      );
      tokenContext.pop();
      return isFunction;
    }

    bool variableLoader_t::isLoadingFunctionPointer() {
      // TODO: Cover the case 'int *()' -> int (*)()'
      const int tokens = tokenContext.size();
      // Function pointer starts with (* or (^
      if (!tokens ||
          !(token_t::safeOperatorType(tokenContext[0]) & operatorType::parenthesesStart)) {
        return false;
      }

      tokenContext.pushPairRange();
      const bool isFunctionPointer = (
        tokenContext.size()
        && (token_t::safeOperatorType(tokenContext[0]) & (operatorType::mult |
                                                          operatorType::xor_))
      );
      tokenContext.pop();
      return isFunctionPointer;
    }

    bool variableLoader_t::loadBasicVariable(vartype_t &vartype,
                                             variable_t &var) {
      identifierToken *nameToken = NULL;
      if (tokenContext.size() &&
          (tokenContext[0]->type() & tokenType::identifier)) {
        nameToken = (identifierToken*) tokenContext[0];
        ++tokenContext;
      }

      setArrays(vartype);
      if (!success) {
        return false;
      }

      var = variable_t(vartype, nameToken);
      return true;
    }

    bool variableLoader_t::loadFunctionPointer(vartype_t &vartype,
                                               variable_t &functionVar) {
      // TODO: Check for nested function pointers
      //       Check for arrays
      tokenContext.pushPairRange();

      functionPtr_t func(vartype);
      func.isBlock = (
        token_t::token_t::safeOperatorType(tokenContext[0]) & operatorType::xor_
      );
      ++tokenContext;

      identifierToken *nameToken = NULL;
      if (tokenContext.size() &&
          (tokenContext[0]->type() & tokenType::identifier)) {
        nameToken = (identifierToken*) tokenContext[0];
        ++tokenContext;
      }

      // If we have arrays, we don't set them in the return type
      vartype_t arraytype;
      setArrays(arraytype);

      if (tokenContext.size()) {
        tokenContext.printError("Unable to parse type");
        return false;
      }

      tokenContext.popAndSkip();

      if (success) {
        tokenContext.pushPairRange();
        setArguments(func);
        if (!success) {
          return false;
        }
        tokenContext.popAndSkip();
      }

      if (!arraytype.arrays.size()) {
        functionVar = variable_t(func, nameToken);
        return true;
      }

      vartype_t varType(func);
      varType.arrays = arraytype.arrays;
      functionVar = variable_t(varType, nameToken);

      return true;
    }

    bool variableLoader_t::loadFunction(function_t &func) {
      success = loadType(tokenContext,
                         smntContext,
                         parser,
                         func.returnType);
      if (!success) {
        return false;
      }

      if (!(token_t::safeType(tokenContext[0]) & tokenType::identifier)) {
        tokenContext.printError("Expected function name identifier");
        return false;
      }
      if (!(token_t::safeOperatorType(tokenContext[1]) & operatorType::parenthesesStart)) {
        tokenContext.printError("Expected parenetheses with function arguments");
        return false;
      }

      func.setSource(tokenContext[0]->to<identifierToken>());
      ++tokenContext;

      tokenContext.pushPairRange();
      setArguments(func);
      tokenContext.popAndSkip();

      const opType_t opType = token_t::safeOperatorType(tokenContext[0]);
      if (!(opType & (operatorType::semicolon |
                      operatorType::braceStart))) {
        tokenContext.printError("[4] Expected a [;]");
        delete &func;
        return false;
      }

      return true;
    }

    bool variableLoader_t::hasArray() {
      return (token_t::safeOperatorType(tokenContext[0]) & operatorType::bracketStart);
    }

    void variableLoader_t::setArrays(vartype_t &vartype) {
      while (success && hasArray()) {
        operatorToken &start = tokenContext[0]->to<operatorToken>();
        operatorToken &end   = tokenContext.getClosingPairToken()->to<operatorToken>();
        tokenContext.pushPairRange();

        exprNode *value = NULL;
        if (tokenContext.size()) {
          value = tokenContext.parseExpression(smntContext, parser);
          success &= !!value;
          if (!success) {
            return;
          }
        }
        vartype += array_t(start, end, value);

        tokenRange pairRange = tokenContext.pop();
        tokenContext += (pairRange.end + 1);
      }
    }

    void variableLoader_t::setArguments(functionPtr_t &func) {
      setArgumentsFor(func);
    }

    void variableLoader_t::setArguments(function_t &func) {
      setArgumentsFor(func);
    }

    void getArgumentRanges(tokenContext_t &tokenContext,
                           tokenRangeVector &argRanges) {
      argRanges.clear();

      tokenContext.push();
      while (true) {
        const int tokens = tokenContext.size();
        if (!tokens) {
          break;
        }

        const int pos = tokenContext.getNextOperator(operatorType::comma);
        // No comma found
        if (pos < 0) {
          argRanges.push_back(tokenRange(0, tokens));
          break;
        }
        argRanges.push_back(tokenRange(0, pos));
        // Trailing comma found
        if (pos == (tokens - 1)) {
          break;
        }
        tokenContext += (pos + 1);
      }
      tokenContext.pop();
    }

    bool loadVariable(tokenContext_t &tokenContext,
                      statementContext_t &smntContext,
                      parser_t &parser,
                      nameToAttributeMap &attributeMap,
                      variable_t &var) {
      variableLoader_t loader(tokenContext, smntContext, parser, attributeMap);
      return loader.loadVariable(var);
    }

    bool loadVariable(tokenContext_t &tokenContext,
                      statementContext_t &smntContext,
                      parser_t &parser,
                      nameToAttributeMap &attributeMap,
                      vartype_t &vartype,
                      variable_t &var) {
      variableLoader_t loader(tokenContext, smntContext, parser, attributeMap);
      return loader.loadVariable(vartype, var);
    }

    bool loadFunction(tokenContext_t &tokenContext,
                      statementContext_t &smntContext,
                      parser_t &parser,
                      nameToAttributeMap &attributeMap,
                      function_t &func) {
      variableLoader_t loader(tokenContext, smntContext, parser, attributeMap);
      return loader.loadFunction(func);
    }

    bool isLoadingVariable(tokenContext_t &tokenContext,
                           statementContext_t &smntContext,
                           parser_t &parser,
                           nameToAttributeMap &attributeMap) {
      variableLoader_t loader(tokenContext, smntContext, parser, attributeMap);
      return loader.isLoadingVariable();
    }

    bool isLoadingFunction(tokenContext_t &tokenContext,
                           statementContext_t &smntContext,
                           parser_t &parser,
                           nameToAttributeMap &attributeMap) {
      variableLoader_t loader(tokenContext, smntContext, parser, attributeMap);
      return loader.isLoadingFunction();
    }

    bool isLoadingFunctionPointer(tokenContext_t &tokenContext,
                                  statementContext_t &smntContext,
                                  parser_t &parser,
                                  nameToAttributeMap &attributeMap) {
      variableLoader_t loader(tokenContext, smntContext, parser, attributeMap);
      return loader.isLoadingFunctionPointer();
    }
  }
}
