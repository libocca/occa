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
#include "expression.hpp"
#include "parser.hpp"
#include "typeBuiltins.hpp"

namespace occa {
  namespace lang {
    parser_t::parser_t() :
      unknownFilter(true),
      root(),
      up(&root) {
      // Properly implement `identifier-nondigit` for identifiers
      // Meanwhile, we use the unknownFilter
      stream = (tokenizer
                .filter(unknownFilter)
                .map(preprocessor)
                .map(stringMerger)
                .map(newlineMerger));

      // Setup simple keyword -> statement peeks
      keywordPeek[keywordType::qualifier]   = statementType::declaration;
      keywordPeek[keywordType::type]        = statementType::declaration;
      keywordPeek[keywordType::if_]         = statementType::if_;
      keywordPeek[keywordType::switch_]     = statementType::switch_;
      keywordPeek[keywordType::case_]       = statementType::case_;
      keywordPeek[keywordType::default_]    = statementType::default_;
      keywordPeek[keywordType::for_]        = statementType::for_;
      keywordPeek[keywordType::while_]      = statementType::while_;
      keywordPeek[keywordType::do_]         = statementType::while_;
      keywordPeek[keywordType::break_]      = statementType::break_;
      keywordPeek[keywordType::continue_]   = statementType::continue_;
      keywordPeek[keywordType::return_]     = statementType::return_;
      keywordPeek[keywordType::public_]     = statementType::classAccess;
      keywordPeek[keywordType::protected_]  = statementType::classAccess;
      keywordPeek[keywordType::private_]    = statementType::classAccess;
      keywordPeek[keywordType::namespace_]  = statementType::namespace_;
      keywordPeek[keywordType::goto_]       = statementType::goto_;

      // Statement type -> loader function
      statementLoaders[statementType::expression]  = &parser_t::loadExpressionStatement;
      statementLoaders[statementType::declaration] = &parser_t::loadDeclarationStatement;
      statementLoaders[statementType::block]       = &parser_t::loadBlockStatement;
      statementLoaders[statementType::namespace_]  = &parser_t::loadNamespaceStatement;
      statementLoaders[statementType::typeDecl]    = &parser_t::loadTypeDeclStatement;
      statementLoaders[statementType::if_]         = &parser_t::loadIfStatement;
      statementLoaders[statementType::elif_]       = &parser_t::loadElifStatement;
      statementLoaders[statementType::else_]       = &parser_t::loadElseStatement;
      statementLoaders[statementType::for_]        = &parser_t::loadForStatement;
      statementLoaders[statementType::while_]      = &parser_t::loadWhileStatement;
      statementLoaders[statementType::switch_]     = &parser_t::loadSwitchStatement;
      statementLoaders[statementType::case_]       = &parser_t::loadCaseStatement;
      statementLoaders[statementType::default_]    = &parser_t::loadDefaultStatement;
      statementLoaders[statementType::continue_]   = &parser_t::loadContinueStatement;
      statementLoaders[statementType::break_]      = &parser_t::loadBreakStatement;
      statementLoaders[statementType::return_]     = &parser_t::loadReturnStatement;
      statementLoaders[statementType::classAccess] = &parser_t::loadClassAccessStatement;
      statementLoaders[statementType::attribute]   = &parser_t::loadAttributeStatement;
      statementLoaders[statementType::pragma]      = &parser_t::loadPragmaStatement;
      statementLoaders[statementType::goto_]       = &parser_t::loadGotoStatement;
      statementLoaders[statementType::gotoLabel]   = &parser_t::loadGotoLabelStatement;
    }

    parser_t::~parser_t() {
      clear();
    }

    //---[ Setup ]----------------------
    void parser_t::clear() {
      tokenizer.clear();
      preprocessor.clear();
      context.clear();

      root.clear();
      up = &root;

      freeKeywords(keywords);
      attributes.clear();

      success = true;
    }

    void parser_t::parseSource(const std::string &source) {
      setSource(source, false);
      if (!success) {
        return;
      }
      parseTokens();
    }

    void parser_t::parseFile(const std::string &filename) {
      setSource(filename, true);
      if (!success) {
        return;
      }
      parseTokens();
    }

    void parser_t::setSource(const std::string &source,
                             const bool isFile) {
      clear();

      if (isFile) {
        tokenizer.set(new file_t(source));
      } else {
        tokenizer.set(source.c_str());
      }

      loadTokens();
    }

    void parser_t::loadTokens() {
      token_t *token;
      while (!stream.isEmpty()) {
        stream >> token;
        context.tokens.push_back(token);
      }

      if (tokenizer.errors ||
          preprocessor.errors) {
        success = false;
        return;
      }

      context.setup();
      success = !context.hasError;

      if (success) {
        getKeywords(keywords);
      }
    }

    void parser_t::parseTokens() {
      loadAllStatements(root.children);
    }

    keyword_t* parser_t::getKeyword(token_t *token) {
      if (!(token_t::safeType(token) & tokenType::identifier)) {
        return NULL;
      }

      identifierToken &identifier = token->to<identifierToken>();
      return keywords.get(identifier.value).value();
    }

    opType_t parser_t::getOperatorType(token_t *token) {
      if (!(token_t::safeType(token) & tokenType::op)) {
        return operatorType::none;
      }
      return token->to<operatorToken>().getOpType();
    }
    //==================================

    //---[ Peek ]-----------------------
    int parser_t::peek() {
      const int tokens = context.size();
      if (!tokens) {
        return statementType::none;
      }

      int tokenIndex = 0;

      while (success &&
             (tokenIndex < tokens)) {

        token_t *token = context[tokenIndex];
        const int tokenType = token->type();

        if (tokenType & tokenType::identifier) {
          return peekIdentifier(tokenIndex);
        }

        if (tokenType & tokenType::op) {
          return peekOperator(tokenIndex);
        }

        if (tokenType & (tokenType::primitive |
                         tokenType::string    |
                         tokenType::char_)) {
          return statementType::expression;
        }

        if (tokenType & tokenType::pragma) {
          return statementType::pragma;
        }

        ++tokenIndex;
      }

      return statementType::none;
    }

    int parser_t::peekIdentifier(const int tokenIndex) {
      token_t *token     = context[tokenIndex];
      keyword_t *keyword = getKeyword(token);

      if (!keyword) {
        // Test for : for it to be a goto label
        if (isGotoLabel(tokenIndex + 1)) {
          return statementType::gotoLabel;
        }
        // TODO: Attempt to find error by guessing the keyword type
        token->printError("Unknown identifier");
        success = false;
        return statementType::none;
      }

      const int kType = keyword->type();
      const int sType = keywordPeek[kType];

      if (sType) {
        return sType;
      }

      if (kType & keywordType::else_) {
        keyword_t *nextKeyword = getKeyword(context[tokenIndex + 1]);
        if (nextKeyword &&
            (nextKeyword->type() & keywordType::if_)) {
          return statementType::elif_;
        }
        return statementType::else_;
      }

      return statementType::expression;
    }

    bool parser_t::isGotoLabel(const int tokenIndex) {
      return (getOperatorType(context[tokenIndex]) & operatorType::colon);
    }

    int parser_t::peekOperator(const int tokenIndex) {
      const opType_t opType = getOperatorType(context[tokenIndex]);
      if (opType & operatorType::braceStart) {
        return statementType::block;
      }
      if (opType & operatorType::attribute) {
        return statementType::attribute;
      }
      return statementType::expression;
    }
    //==================================

    //---[ Type Loaders ]---------------
    variable parser_t::loadVariable() {
      vartype_t vartype = preloadType();
      if (isLoadingFunctionPointer()) {
        return loadFunctionPointer(vartype);
      }
      return loadVariable(vartype);
    }

    variableDeclaration parser_t::loadVariableDeclaration(const vartype_t &baseType) {
      variableDeclaration decl;

      // If partially-defined type, finish parsing it
      vartype_t vartype = baseType.declarationType();
      vartype.qualifiers = baseType.qualifiers;
      setPointers(vartype);
      if (!success) {
        return decl;
      }
      setReference(vartype);
      if (!success) {
        return decl;
      }
      // Load the actual variable if it exists
      if (isLoadingFunctionPointer()) {
        decl.var = loadFunctionPointer(vartype);
      } else {
        decl.var = loadVariable(vartype);
      }

      if (!context.size() ||
          !(getOperatorType(context[0]) & operatorType::assign)) {
        return decl;
      }

      const int pos = context.getNextOperator(operatorType::comma |
                                              operatorType::semicolon);
      if (pos < 0) {
        context.end()->printError("Expected a ;");
        success = false;
        return decl;
      }
      if (pos == 1) {
        context[1]->printError("Expected an expression");
        success = false;
        return decl;
      }

      decl.value = context.getExpression(1, pos);
      context.set(pos);

      return decl;
    }

    vartype_t parser_t::preloadType() {
      // TODO: Handle weird () cases:
      //        int (*const (*const a))      -> int * const * const a;
      //        int (*const (*const (*a)))() -> int (* const * const *a)();
      // Set the name in loadBaseType and look for (*)() or (^)()
      //   to stop qualifier merging
      vartype_t vartype;
      if (!context.size()) {
        return vartype;
      }

      loadBaseType(vartype);
      if (!success ||
          !vartype.isValid()) {
        return vartype;
      }

      setPointers(vartype);
      if (!success) {
        return vartype;
      }

      setReference(vartype);
      return vartype;
    }

    void parser_t::loadBaseType(vartype_t &vartype) {
      const int tokens = context.size();
      int tokenPos;
      for (tokenPos = 0; tokenPos < tokens; ++tokenPos) {
        token_t *token     = context[tokenPos];
        keyword_t *keyword = getKeyword(token);
        if (!keyword) {
          break;
        }

        const int kType = keyword->type();
        if (kType & keywordType::qualifier) {
          loadQualifier(token,
                        keyword->to<qualifierKeyword>().qualifier,
                        vartype);
          continue;
        }
        if ((kType & keywordType::type) &&
            !vartype.isValid()) {
          vartype.type = &(keyword->to<typeKeyword>().type_);
          continue;
        }
        break;
      }

      if (tokenPos == 0) {
        context[0]->printError("Unable to load type");
        success = false;
        return;
      }

      // Store token just in case we didn't load a type
      token_t *lastToken = context[tokenPos - (tokenPos == tokens)];
      context.set(tokenPos);

      if (vartype.isValid()) {
        return;
      }

      if (vartype.has(long_) ||
          vartype.has(longlong_)) {
        vartype.type = &int_;
        return;
      }

      lastToken->printError("Expected a type");
      success = false;
    }

    void parser_t::loadQualifier(token_t *token,
                                 const qualifier_t &qualifier,
                                 vartype_t &vartype) {
      // Handle long/long long case
      if (&qualifier == &long_) {
        if (vartype.has(long_)) {
          vartype -= long_;
          vartype.add(token->origin,
                      longlong_);
        }
        else if (vartype.has(longlong_)) {
          token->printWarning("'long long long' is tooooooo long,"
                              " ignoring additional longs");
        }
        else {
          vartype.add(token->origin,
                      long_);
        }
        return;
      }

      // Non-long qualifiers
      if (!vartype.has(qualifier)) {
        vartype.add(token->origin,
                    qualifier);
      } else {
        token->printWarning("Ignoring duplicate qualifier");
      }
    }

    void parser_t::setPointers(vartype_t &vartype) {
      while (success &&
             context.size()) {
        if (!(getOperatorType(context[0]) & operatorType::mult)) {
          break;
        }
        context.set(1);
        setPointer(vartype);
      }
    }

    void parser_t::setPointer(vartype_t &vartype) {
      pointer_t pointer;

      const int tokens = context.size();
      int tokenPos;
      for (tokenPos = 0; tokenPos < tokens; ++tokenPos) {
        token_t *token     = context[tokenPos];
        keyword_t *keyword = getKeyword(token);
        if (!(keyword_t::safeType(keyword) & keywordType::qualifier)) {
          break;
        }

        const qualifier_t &qualifier = keyword->to<qualifierKeyword>().qualifier;
        if (!(qualifier.type() & qualifierType::forPointers)) {
          token->printError("Cannot add this qualifier to a pointer");
          success = false;
          break;
        }
        pointer.add(token->origin,
                    qualifier);
      }

      context.set(tokenPos);

      if (success) {
        vartype += pointer;
      }
    }

    void parser_t::setReference(vartype_t &vartype) {
      if (!context.size()) {
        return;
      }
      if (!(getOperatorType(context[0]) & operatorType::bitAnd)) {
        return;
      }
      vartype.setReferenceToken(context[0]);
      context.set(1);
    }

    bool parser_t::isLoadingFunctionPointer() {
      // TODO: Cover the case 'int *()' -> int (*)()'
      const int tokens = context.size();
      // Function pointer starts with (* or (^
      if (!tokens ||
          !(getOperatorType(context[0]) & operatorType::parenthesesStart)) {
        return false;
      }

      context.pushPairRange(0);
      const bool isFunctionPointer = (
        context.size()
        && (getOperatorType(context[0]) & (operatorType::mult |
                                           operatorType::xor_))
      );
      context.pop();
      return isFunctionPointer;
    }

    bool parser_t::isLoadingVariable() {
      const int tokens = context.size();
      // Variable must have an identifier token next
      if (!tokens ||
          (!(context[0]->type() & tokenType::identifier))) {
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
      return (!(getOperatorType(context[1]) & operatorType::parenthesesStart) ||
              (getOperatorType(context[2]) & operatorType::parenthesesStart));
    }

    variable parser_t::loadFunctionPointer(vartype_t &vartype) {
      // TODO: Check for nested function pointers
      //       Check for arrays
      context.pushPairRange(0);

      identifierToken *nameToken = NULL;
      const bool isPointer = (getOperatorType(context[0]) & operatorType::mult);
      context.set(1);

      function_t func(vartype);
      func.isPointer = isPointer;
      func.isBlock   = !isPointer;

      if (context.size() &&
          (context[0]->type() & tokenType::identifier)) {
        nameToken = (identifierToken*) context[0];
        context.set(1);
      }

      // If we have arrays, we don't set them in the return type
      vartype_t arraytype;
      setArrays(arraytype);

      if (context.size()) {
        context[0]->printError("Unable to parse type");
        success = false;
      }

      context.popAndSkipPair();

      if (success) {
        context.pushPairRange(0);
        setArguments(func.args);
        context.popAndSkipPair();
      }

      if (!arraytype.arrays.size()) {
        return variable(func, nameToken);
      }

      vartype_t varType(func);
      varType.arrays = arraytype.arrays;
      return variable(varType, nameToken);
    }

    variable parser_t::loadVariable(vartype_t &vartype) {
      identifierToken *nameToken = NULL;
      if (context.size() &&
          (context[0]->type() & tokenType::identifier)) {
        nameToken = (identifierToken*) context[0];
        context.set(1);
      }

      setArrays(vartype);

      return variable(vartype, nameToken);
    }

    bool parser_t::hasArray() {
      return (context.size() &&
              (getOperatorType(context[0]) & operatorType::bracketStart));
    }

    void parser_t::setArrays(vartype_t &vartype) {
      while (success &&
             hasArray()) {

        operatorToken &start = context[0]->to<operatorToken>();
        operatorToken &end   = context.getClosingPairToken(0)->to<operatorToken>();
        context.pushPairRange(0);

        vartype += array_t(start,
                           end,
                           context.getExpression());

        tokenRange pairRange = context.pop();
        context.set(pairRange.end + 1);
      }
    }

    void parser_t::setArguments(variableVector &args) {
      tokenRangeVector argRanges;
      getArgumentRanges(argRanges);

      const int argCount = (int) argRanges.size();
      if (!argCount) {
        return;
      }

      for (int i = 0; i < argCount; ++i) {
        context.push(argRanges[i].start,
                     argRanges[i].end);

        args.push_back(loadVariable());
        if (!success) {
          break;
        }

        context.pop();
        context.set(argRanges[i].end + 1);
      }
    }

    void parser_t::getArgumentRanges(tokenRangeVector &argRanges) {
      argRanges.clear();

      context.push();
      while (true) {
        const int tokens = context.size();
        if (!tokens) {
          break;
        }

        const int pos = context.getNextOperator(operatorType::comma);
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
        context.set(pos + 1);
      }
      context.pop();
    }

    class_t parser_t::loadClassType() {
      context[0]->printError("Cannot parse classes yet");
      success = false;
      return class_t();
    }

    struct_t parser_t::loadStructType() {
      context[0]->printError("Cannot parse structs yet");
      success = false;
      return struct_t();
    }

    enum_t parser_t::loadEnumType() {
      context[0]->printError("Cannot parse enum yet");
      success = false;
      return enum_t();
    }

    union_t parser_t::loadUnionType() {
      context[0]->printError("Cannot parse union yet");
      success = false;
      return union_t();
    }
    //==================================

    //---[ Statement Loaders ]----------
    void parser_t::loadAllStatements(statementPtrVector &statements) {
      statement_t *smnt = getNextStatement();
      while (smnt) {
        statements.push_back(smnt);
        smnt = getNextStatement();
      }

      if (!success) {
        const int count = (int) statements.size();
        for (int i = 0; i < count; ++i) {
          delete statements[i];
        }
        statements.clear();
      }
    }

    statement_t* parser_t::getNextStatement() {
      const int sType = peek();
      if ((!success) ||
          (sType & (statementType::none |
                    statementType::empty))) {
        return NULL;
      }

      statementLoaderMap::iterator it = statementLoaders.find(sType);
      if (it != statementLoaders.end()) {
        statementLoader_t loader = it->second;
        statement_t *smnt = (this->*loader)();
        if (!success) {
          delete smnt;
          smnt = NULL;
        }
        return smnt;
      }

      OCCA_FORCE_ERROR("[Waldo] Oops, forgot to implement a statement loader"
                       " for [" << stringifySetBits(sType) << "]");
      return NULL;
    }

    statement_t* parser_t::loadBlockStatement() {
      blockStatement *smnt = new blockStatement();
      loadAllStatements(smnt->children);
      return smnt;
    }

    statement_t* parser_t::loadExpressionStatement() {
      const int end = context.getNextOperator(operatorType::semicolon);
      if (end < 0) {
        context.end()->printError("Missing ;");
        success = false;
        return NULL;
      }

      context.push(0, end);

      tokenVector tokens;
      context.getAndCloneTokens(tokens);

      exprNode *expr = getExpression(tokens);
      if (!expr) {
        success = false;
        return NULL;
      }

      context.pop();
      context.set(end + 1);

      return new expressionStatement(*expr);
    }

    statement_t* parser_t::loadDeclarationStatement() {
      vartype_t baseType = preloadType();
      if (!success) {
        return NULL;
      }

      declarationStatement &smnt = *(new declarationStatement());
      while(success) {
        smnt.declarations.push_back(loadVariableDeclaration(baseType));
        if (!success) {
          break;
        }

        // Skip the , or ;
        if (!(getOperatorType(context[0]) & operatorType::comma)) {
          context.set(1);
          break;
        }
        context.set(1);
      }

      if (!success) {
        delete &smnt;
        return NULL;
      }

      // Skip the ;
      context.set(1);

      return &smnt;
    }

    statement_t* parser_t::loadNamespaceStatement() {
      return NULL;
    }

    statement_t* parser_t::loadTypeDeclStatement() {
      return NULL;
    }

    statement_t* parser_t::loadIfStatement() {
      return NULL;
    }

    statement_t* parser_t::loadElifStatement() {
      return NULL;
    }

    statement_t* parser_t::loadElseStatement() {
      return NULL;
    }

    statement_t* parser_t::loadForStatement() {
      return NULL;
    }

    statement_t* parser_t::loadWhileStatement() {
      return NULL;
    }

    statement_t* parser_t::loadSwitchStatement() {
      return NULL;
    }

    statement_t* parser_t::loadCaseStatement() {
      const int pos = context.getNextOperator(operatorType::colon);
      // No : found
      if (pos < 0) {
        context[0]->printError("Expected a : to close the case statement");
        success = false;
        return NULL;
      }
      // The case where we see 'case:'
      if (pos == 1) {
        context[1]->printError("Expected a constant expression for the case statement");
        success = false;
        return NULL;
      }

      // Load the case expression
      exprNode *value = context.getExpression(1, pos);
      if (!value) {
        success = false;
        return NULL;
      }

      context.set(pos + 1);
      return new caseStatement(*value);
    }

    statement_t* parser_t::loadDefaultStatement() {
      return NULL;
    }

    statement_t* parser_t::loadContinueStatement() {
      return NULL;
    }

    statement_t* parser_t::loadBreakStatement() {
      return NULL;
    }

    statement_t* parser_t::loadReturnStatement() {
      return NULL;
    }

    statement_t* parser_t::loadClassAccessStatement() {
      return NULL;
    }

    statement_t* parser_t::loadAttributeStatement() {
      return NULL;
    }

    statement_t* parser_t::loadPragmaStatement() {
      pragmaStatement *smnt = new pragmaStatement(context[0]
                                                  ->clone()
                                                  ->to<pragmaToken>());
      context.set(1);
      return smnt;
    }

    statement_t* parser_t::loadGotoStatement() {
      return NULL;
    }

    statement_t* parser_t::loadGotoLabelStatement() {
      return NULL;
    }
    //==================================
  }
}
