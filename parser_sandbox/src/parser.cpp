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
#include "attribute.hpp"
#include "expression.hpp"
#include "parser.hpp"
#include "variable.hpp"
#include "builtins/attributes.hpp"
#include "builtins/transforms.hpp"
#include "builtins/types.hpp"

namespace occa {
  namespace lang {
    identifierReplacer_t::identifierReplacer_t(parser_t &parser_) :
      parser(parser_),
      scopeSmnt(NULL) {}

    exprNode* identifierReplacer_t::transformExprNode(exprNode &node) {
      return &node;
    }

    parser_t::parser_t() :
      unknownFilter(true),
      lastPeek(0),
      lastPeekPosition(-1),
      checkSemicolon(true),
      root(NULL, NULL),
      up(&root),
      identifierReplacer(*this) {
      // Manually set source to avoid a dangling token pointer
      root.source = new newlineToken(filePosition());

      // Properly implement `identifier-nondigit` for identifiers
      // Meanwhile, we use the unknownFilter
      stream = (tokenizer
                .filter(unknownFilter)
                .map(preprocessor)
                .map(stringMerger)
                .map(newlineFilter));

      // Setup simple keyword -> statement peeks
      keywordPeek[keywordType::qualifier]   = statementType::declaration;
      keywordPeek[keywordType::type]        = statementType::declaration;
      keywordPeek[keywordType::variable]    = statementType::expression;
      keywordPeek[keywordType::function]    = statementType::expression;
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
      statementLoaders[statementType::empty]       = &parser_t::loadEmptyStatement;
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
      statementLoaders[statementType::pragma]      = &parser_t::loadPragmaStatement;
      statementLoaders[statementType::goto_]       = &parser_t::loadGotoStatement;
      statementLoaders[statementType::gotoLabel]   = &parser_t::loadGotoLabelStatement;

      addAttribute<attributes::dim>();
      addAttribute<attributes::dimOrder>();
      addAttribute<attributes::tile>();
    }

    parser_t::~parser_t() {
      clear();
      nameToAttributeMap::iterator it = attributeMap.begin();
      while (it != attributeMap.end()) {
        delete it->second;
        ++it;
      }
      attributeMap.clear();
    }

    //---[ Setup ]----------------------
    void parser_t::clear() {
      tokenizer.clear();
      preprocessor.clear();
      context.clear();

      lastPeek = 0;
      lastPeekPosition = -1;
      checkSemicolon = true;

      root.clear();
      up = &root;
      upStack.clear();

      freeKeywords(keywords);
      clearAttributes();

      success = true;
    }

    void parser_t::clearAttributes() {
      clearAttributes(attributes);
    }

    void parser_t::clearAttributes(attributeTokenMap &attrs) {
      attributeTokenMap::iterator it = attrs.begin();
      while (it != attrs.end()) {
        it->second.clear();
        ++it;
      }
      attrs.clear();
    }

    void parser_t::pushUp(blockStatement &newUp) {
      upStack.push_back(up);
      up = &newUp;
    }

    void parser_t::popUp() {
      if (upStack.size()) {
        up = upStack.back();
        upStack.pop_back();
      } else {
        up = &root;
      }
    }

    void parser_t::parseSource(const std::string &source) {
      setSource(source, false);
      if (success) {
        parseTokens();
      }
    }

    void parser_t::parseFile(const std::string &filename) {
      setSource(filename, true);
      if (success) {
        parseTokens();
      }
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
      loadAllStatements();
      if (!success) return;
      success = transforms::applyDimTransforms(root);
      if (!success) return;
      success = transforms::applyTileTransforms(root);
    }

    keyword_t& parser_t::getKeyword(token_t *token) {
      static keyword_t noKeyword;

      if (!(token_t::safeType(token) & tokenType::identifier)) {
        return noKeyword;
      }

      std::string &identifier = token->to<identifierToken>().value;
      keywordMapIterator it = keywords.find(identifier);
      if (it != keywords.end()) {
        return *(it->second);
      }

      return up->getScopeKeyword(identifier);
    }

    opType_t parser_t::getOperatorType(token_t *token) {
      if (!(token_t::safeType(token) & tokenType::op)) {
        return operatorType::none;
      }
      return token->to<operatorToken>().getOpType();
    }
    //==================================

    //---[ Helper Methods ]-------------
    exprNode* parser_t::getExpression() {
      return getExpression(0, context.size());
    }

    exprNode* parser_t::getExpression(const int start,
                                      const int end) {
      if (!up) {
        return context.getExpression(start, end);
      }

      for (int i = start; i < end; ++i) {
        token_t *token = context[i];
        if (token->type() & tokenType::identifier) {
          context.setToken(i,
                           replaceIdentifier((identifierToken&) *token));
        }
      }
      return context.getExpression(start, end);
    }

    token_t* parser_t::replaceIdentifier(identifierToken &identifier) {
      keyword_t &keyword = up->getScopeKeyword(identifier.value);
      const int kType = keyword.type();
      if (!(kType & (keywordType::type     |
                     keywordType::variable |
                     keywordType::function))) {
        return &identifier;
      }

      if (kType & keywordType::variable) {
        return new variableToken(identifier.origin,
                                 ((variableKeyword&) keyword).variable);
      }
      if (kType & keywordType::function) {
        return new functionToken(identifier.origin,
                                 ((functionKeyword&) keyword).function);
      }
      // keywordType::type
      return new typeToken(identifier.origin,
                           ((typeKeyword&) keyword).type_);
    }

    void parser_t::loadAttributes(attributeTokenMap &attrs) {
      while (success &&
             (getOperatorType(context[0]) & operatorType::attribute)) {
        loadAttribute(attrs);
        if (!success) {
          break;
        }
      }
    }

    void parser_t::loadAttribute(attributeTokenMap &attrs) {
      // Skip [@] token
      context.set(1);

      if (!(context[0]->type() & tokenType::identifier)) {
        context.printError("Expected a namespace name");
        success = false;
        return;
      }

      identifierToken &nameToken = *((identifierToken*) context[0]);
      context.set(1);

      nameToAttributeMap::iterator it = attributeMap.find(nameToken.value);
      if (it == attributeMap.end()) {
        nameToken.printError("Unknown attribute");
        success = false;
        return;
      }
      attribute_t &attribute = *(it->second);

      tokenRangeVector argRanges;
      const bool hasArgs = (getOperatorType(context[0]) & operatorType::parenthesesStart);
      if (hasArgs) {
        context.pushPairRange(0);
        getArgumentRanges(argRanges);
      }
      if (!success) {
        if (hasArgs) {
          context.popAndSkip();
        }
        return;
      }

      attributeToken_t attr(*(it->second), nameToken);
      setAttributeArgs(attr, argRanges);
      if (!success) {
        context.popAndSkip();
        return;
      }

      attrs[nameToken.value] = attr;

      if (hasArgs) {
        context.popAndSkip();
      }

      success = attribute.isValid(attr);
    }

    void parser_t::setAttributeArgs(attributeToken_t &attr,
                                    tokenRangeVector &argRanges) {
      const int args = (int) argRanges.size();
      bool foundNamedArg = false;
      for (int i = 0; i < args; ++i) {
        context.push(argRanges[i].start,
                     argRanges[i].end);

        if (!context.size()) {
          attr.args.push_back(new emptyNode());
          context.popAndSkip();
          continue;
        }

        // Load args
        attributeTokenMap attrs;
        loadAttributes(attrs);

        if (!context.size()) {
          attr.args.push_back(
            attributeArg_t(new emptyNode(),
                           attrs)
          );
          context.popAndSkip();
          continue;
        }

        // Get argument
        exprNode *arg = getExpression();
        if (!arg) {
          success = false;
          context.pop();
          return;
        }

        std::string argName;
        // Check for
        // |---[=] (binary)
        // |   |
        // |   |---[argName] (identifier)
        // |   |
        // |   |---[arg] (?)
        if (arg->type() & exprNodeType::binary) {
          binaryOpNode &equalsNode = arg->to<binaryOpNode>();
          if ((equalsNode.opType() & operatorType::assign) &&
              (equalsNode.leftValue->type() & exprNodeType::identifier)) {
            argName = equalsNode.leftValue->to<identifierNode>().value;
            arg = equalsNode.rightValue->clone();
            delete &equalsNode;
          }
        }

        if (!argName.size() &&
            foundNamedArg) {
          context.printError("All arguments after a named argument"
                             " must also be named");
          success = false;
          delete arg;
          context.pop();
          return;
        }

        if (!argName.size()) {
          attr.args.push_back(arg);
        } else {
          attr.kwargs[argName] = arg;
        }

        context.popAndSkip();
      }

    }

    void parser_t::addAttributesTo(attributeTokenMap &attrs,
                                   statement_t *smnt) {
      if (!smnt) {
        clearAttributes(attrs);
        return;
      }

      const int sType = smnt->type();
      attributeTokenMap::iterator it = attrs.begin();
      while (it != attrs.end()) {
        attributeToken_t &attr = it->second;
        if (attr.forStatement(sType)) {
          smnt->addAttribute(attr);
        } else {
          attr.printError("Cannot apply attribute to this type of statement");
          smnt->attributes.clear();
          clearAttributes(attrs);
          success = false;
          break;
        }
        ++it;
      }
      attrs.clear();
    }
    //==================================

    //---[ Peek ]-----------------------
    int parser_t::peek() {
      const int contextPosition = context.position();
      if (lastPeekPosition != contextPosition) {
        setupPeek();
        lastPeek = (success
                    ? uncachedPeek()
                    : statementType::none);
        lastPeekPosition = contextPosition;
      }
      return lastPeek;
    }

    int parser_t::uncachedPeek() {
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

    void parser_t::setupPeek() {
      int contextPos = -1;
      while (success &&
             context.size() &&
             (contextPos != context.position())) {
        contextPos = context.position();
        loadAttributes(attributes);
      }
    }

    int parser_t::peekIdentifier(const int tokenIndex) {
      token_t *token = context[tokenIndex];
      int kType = getKeyword(token).type();

      if (kType & keywordType::none) {
        // Test for : for it to be a goto label
        if (isGotoLabel(tokenIndex + 1)) {
          return statementType::gotoLabel;
        }
        // TODO: Make sure it's a defined variable
        return statementType::expression;
      }

      const int sType = keywordPeek[kType];
      if (sType) {
        return sType;
      }

      if (kType & keywordType::else_) {
        keyword_t &nextKeyword = getKeyword(context[tokenIndex + 1]);
        if ((nextKeyword.type() & keywordType::if_)) {
          return statementType::elif_;
        }
        return statementType::else_;
      }

      token->printError("Unknown identifier");
      success = false;
      return statementType::none;
    }

    bool parser_t::isGotoLabel(const int tokenIndex) {
      return (getOperatorType(context[tokenIndex]) & operatorType::colon);
    }

    int parser_t::peekOperator(const int tokenIndex) {
      const opType_t opType = getOperatorType(context[tokenIndex]);
      if (opType & operatorType::braceStart) {
        return statementType::block;
      }
      if (opType & operatorType::semicolon) {
        return statementType::empty;
      }
      return statementType::expression;
    }
    //==================================

    //---[ Type Loaders ]---------------
    variable_t parser_t::loadVariable() {
      attributeTokenMap attrs;
      loadAttributes(attrs);

      vartype_t vartype = preloadType();
      variable_t var = (!isLoadingFunctionPointer()
                        ? loadVariable(vartype)
                        : loadFunctionPointer(vartype));

      if (var.vartype.type) {
        var.attributes = var.vartype.type->attributes;
      }
      var.attributes.insert(attrs.begin(), attrs.end());

      return var;
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
      decl.variable = &(!isLoadingFunctionPointer()
                        ? loadVariable(vartype).clone()
                        : loadFunctionPointer(vartype).clone());

      loadDeclarationAttributes(decl);
      if (!success) {
        return decl;
      }

      loadDeclarationBitfield(decl);
      loadDeclarationAssignment(decl);
      if (!decl.value) {
        loadDeclarationBraceInitializer(decl);
      }

      return decl;
    }


    void parser_t::loadDeclarationAttributes(variableDeclaration &decl) {
      attributeTokenMap &varAttributes = decl.variable->attributes;
      // Copy statement attributes to each variable
      // Variable attributes should override statement attributes
      varAttributes.insert(attributes.begin(),
                           attributes.end());
      loadAttributes(varAttributes);
      if (!success) {
        return;
      }

      attributeTokenMap::iterator it = varAttributes.begin();
      while (it != varAttributes.end()) {
        attributeToken_t &attr = it->second;
        if (!attr.forVariable()) {
          attr.printError("Cannot apply attribute to variables");
          success = false;
        }
        ++it;
      }
    }

    int parser_t::declarationNextCheck(const opType_t opCheck) {
      int pos = context.getNextOperator(opCheck);
      if (pos < 0) {
        if (checkSemicolon) {
          context.printErrorAtEnd("Expected a [;]");
          success = false;
        }
        pos = context.size();
      }
      return pos;
    }

    void parser_t::loadDeclarationBitfield(variableDeclaration &decl) {
      if (!(getOperatorType(context[0]) & operatorType::colon)) {
        return;
      }

      int pos = declarationNextCheck(operatorType::assign     |
                                     operatorType::braceStart |
                                     operatorType::comma      |
                                     operatorType::semicolon);
      if (pos == 1) {
        context[1]->printError("Expected an expression");
        success = false;
      }
      if (!success) {
        return;
      }

      exprNode *value = getExpression(1, pos);
      decl.variable->vartype.bitfield = (int) value->evaluate();
      delete value;
      context.set(pos);
    }

    void parser_t::loadDeclarationAssignment(variableDeclaration &decl) {
      if (!(getOperatorType(context[0]) & operatorType::assign)) {
        return;
      }

      int pos = declarationNextCheck(operatorType::comma |
                                     operatorType::semicolon);
      if (pos == 1) {
        context[1]->printError("Expected an expression");
        success = false;
      }
      if (!success) {
        return;
      }

      decl.value = getExpression(1, pos);
      context.set(pos);
    }

    // TODO: Store brace initializer propertly, maybe in value with an extra flag
    void parser_t::loadDeclarationBraceInitializer(variableDeclaration &decl) {
      if (!(getOperatorType(context[0]) & operatorType::braceStart)) {
        return;
      }

      context.push(context.getClosingPair(0) + 1);
      int pos = declarationNextCheck(operatorType::comma |
                                     operatorType::semicolon);
      if ((pos != 0) &&
          (pos != context.size())) {
        context.printError("Expected a [,] for another variable"
                           " or a stopping [;]");
        success = false;
      }
      if (!success) {
        return;
      }
      context.pop();

      context.pushPairRange(0);
      decl.value = getExpression();
      context.popAndSkip();
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
      // Type was already loaded
      if (vartype.type) {
        return;
      }

      const int tokens = context.size();
      int tokenPos;
      for (tokenPos = 0; tokenPos < tokens; ++tokenPos) {
        token_t *token     = context[tokenPos];
        keyword_t &keyword = getKeyword(token);
        const int kType    = keyword.type();
        if (kType & keywordType::none) {
          break;
        }

        if (kType & keywordType::qualifier) {
          loadQualifier(token,
                        keyword.to<qualifierKeyword>().qualifier,
                        vartype);
          continue;
        }
        if ((kType & keywordType::type) &&
            !vartype.isValid()) {
          vartype.type = &(keyword.to<typeKeyword>().type_);
          continue;
        }
        break;
      }

      if (tokenPos == 0) {
        context.printError("Unable to load type");
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
        keyword_t &keyword = getKeyword(token);
        if (!(keyword.type() & keywordType::qualifier)) {
          break;
        }

        const qualifier_t &qualifier = keyword.to<qualifierKeyword>().qualifier;
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

    bool parser_t::isLoadingFunction() {
      context.push();

      vartype_t vartype = preloadType();
      if (!success) {
        context.popAndSkip();
        return false;
      }

      if (!(token_t::safeType(context[0]) & tokenType::identifier)) {
        context.popAndSkip();
        return false;
      }

      const bool isFunction = (getOperatorType(context[1])
                               & operatorType::parenthesesStart);
      context.pop();
      return isFunction;
    }

    variable_t parser_t::loadFunctionPointer(vartype_t &vartype) {
      // TODO: Check for nested function pointers
      //       Check for arrays
      context.pushPairRange(0);

      functionPtr_t func(vartype);
      func.isBlock = (getOperatorType(context[0]) & operatorType::xor_);
      context.set(1);

      identifierToken *nameToken = NULL;
      if (context.size() &&
          (context[0]->type() & tokenType::identifier)) {
        nameToken = (identifierToken*) context[0];
        context.set(1);
      }

      // If we have arrays, we don't set them in the return type
      vartype_t arraytype;
      setArrays(arraytype);

      if (context.size()) {
        context.printError("Unable to parse type");
        success = false;
      }

      context.popAndSkip();

      if (success) {
        context.pushPairRange(0);
        setArguments(func);
        context.popAndSkip();
      }

      if (!arraytype.arrays.size()) {
        return variable_t(func, nameToken);
      }

      vartype_t varType(func);
      varType.arrays = arraytype.arrays;
      return variable_t(varType, nameToken);
    }

    variable_t parser_t::loadVariable(vartype_t &vartype) {
      identifierToken *nameToken = NULL;
      if (context.size() &&
          (context[0]->type() & tokenType::identifier)) {
        nameToken = (identifierToken*) context[0];
        context.set(1);
      }

      setArrays(vartype);

      return variable_t(vartype, nameToken);
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
                           getExpression());

        tokenRange pairRange = context.pop();
        context.set(pairRange.end + 1);
      }
    }

    void parser_t::setArguments(functionPtr_t &func) {
      setArgumentsFor(func);
    }

    void parser_t::setArguments(function_t &func) {
      setArgumentsFor(func);
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
      context.printError("Cannot parse classes yet");
      success = false;
      return class_t();
    }

    struct_t parser_t::loadStructType() {
      context.printError("Cannot parse structs yet");
      success = false;
      return struct_t();
    }

    enum_t parser_t::loadEnumType() {
      context.printError("Cannot parse enum yet");
      success = false;
      return enum_t();
    }

    union_t parser_t::loadUnionType() {
      context.printError("Cannot parse union yet");
      success = false;
      return union_t();
    }
    //==================================

    //---[ Loader Helpers ]-------------
    bool parser_t::isEmpty() {
      const int sType = peek();
      return (!success ||
              (sType & statementType::none));
    }

    statement_t* parser_t::getNextStatement() {
      if (isEmpty()) {
        checkSemicolon = true;
        return NULL;
      }

      const int sType = peek();

      statementLoaderMap::iterator it = statementLoaders.find(sType);
      if (it != statementLoaders.end()) {
        statementLoader_t loader = it->second;
        statement_t *smnt = (this->*loader)();
        if (!smnt) {
          return NULL;
        }
        if (!success) {
          delete smnt;
          return NULL;
        }

        // [checkSemicolon] is only valid for one statement
        checkSemicolon = true;
        addAttributesTo(attributes, smnt);
        if (!success) {
          delete smnt;
          return NULL;
        }

        return smnt;
      }

      OCCA_FORCE_ERROR("[Waldo] Oops, forgot to implement a statement loader"
                       " for [" << stringifySetBits(sType) << "]");
      return NULL;
    }
    //==================================

    //---[ Statement Loaders ]----------
    void parser_t::loadAllStatements() {
      statementPtrVector &statements = up->children;
      statement_t *smnt = getNextStatement();

      while (smnt) {
        statements.push_back(smnt);
        smnt = getNextStatement();
      }
    }

    statement_t* parser_t::loadBlockStatement() {
      blockStatement *smnt = new blockStatement(up, context[0]);
      addAttributesTo(attributes, smnt);

      context.pushPairRange(0);
      pushUp(*smnt);
      loadAllStatements();
      popUp();
      context.popAndSkip();
      if (!success) {
        delete smnt;
        return NULL;
      }

      return smnt;
    }

    statement_t* parser_t::loadEmptyStatement() {
      statement_t *smnt = new emptyStatement(up, context[0]);
      // Skip [;] token
      context.set(1);
      return smnt;
    }

    statement_t* parser_t::loadExpressionStatement() {
      int end = context.getNextOperator(operatorType::semicolon);
      if (end < 0) {
        if (checkSemicolon) {
          context.printErrorAtEnd("Expected a [;]");
          success = false;
          return NULL;
        }
        end = context.size();
      }

      context.push(0, end);
      exprNode *expr = getExpression();
      context.pop();
      if (!expr) {
        success = false;
        return NULL;
      }
      context.set(end + 1);

      return new expressionStatement(up, *expr);
    }

    statement_t* parser_t::loadDeclarationStatement() {
      if (isLoadingFunction()) {
        return loadFunctionStatement();
      }
      if (!success) {
        return NULL;
      }

      vartype_t baseType;
      loadBaseType(baseType);
      if (!success) {
        return NULL;
      }

      declarationStatement &smnt = *(new declarationStatement(up));
      while(success) {
        // TODO: Pass decl as argument to prevent a copy
        success = smnt.addDeclaration(
          loadVariableDeclaration(baseType)
        );
        if (!success) {
          break;
        }
        const opType_t opType = getOperatorType(context[0]);
        if (!(opType & operatorType::comma)) {
          if (opType & operatorType::semicolon) {
            context.set(1);
          } else if (checkSemicolon) {
            context.printErrorAtEnd("Expected a [;]");
            success = false;
          }
          break;
        }
        context.set(1);
      }
      if (!success) {
        smnt.freeDeclarations();
        delete &smnt;
        return NULL;
      }
      return &smnt;
    }

    statement_t* parser_t::loadNamespaceStatement() {
      if (context.size() == 1) {
        context.printError("Expected a namespace name");
        return NULL;
      }

      // Skip [namespace] token
      context.set(1);
      tokenVector names;

      while (true) {
        // Get the namespace name
        if (!(context[0]->type() & tokenType::identifier)) {
          context.printError("Expected a namespace name");
          success = false;
          return NULL;
        }
        names.push_back(context[0]);

        // Check we still have a token for {
        if (context.size() == 1) {
          context.printError("Missing namespace body {}");
          success = false;
          return NULL;
        }
        context.set(1);

        // Find { or ::
        const opType_t opType = getOperatorType(context[0]);
        if (!(opType & (operatorType::braceStart |
                        operatorType::scope))) {
          context.printError("Expected namespace body {}");
          success = false;
          return NULL;
        }

        if (opType & operatorType::braceStart) {
          break;
        }
        if (context.size() == 1) {
          context.printError("Missing namespace body {}");
          success = false;
          return NULL;
        }
        context.set(1);
      }

      namespaceStatement *smnt = NULL;
      namespaceStatement *currentSmnt = NULL;

      const int levels = (int) names.size();
      for (int i = 0; i < levels; ++i) {
        namespaceStatement *nextSmnt = new namespaceStatement(up,
                                                              names[i]
                                                              ->clone()
                                                              ->to<identifierToken>());
        if (!smnt) {
          smnt = nextSmnt;
          currentSmnt = nextSmnt;
        } else {
          currentSmnt->add(*nextSmnt);
          currentSmnt = nextSmnt;
        }
      }

      addAttributesTo(attributes, currentSmnt);

      // Load block content
      context.pushPairRange(0);
      pushUp(*currentSmnt);
      loadAllStatements();
      popUp();
      context.popAndSkip();

      return smnt;
    }

    statement_t* parser_t::loadTypeDeclStatement() {
      return NULL;
    }

    statement_t* parser_t::loadFunctionStatement() {
      vartype_t returnType = preloadType();

      if (!(token_t::safeType(context[0]) & tokenType::identifier)) {
        context.printError("Expected function name identifier");
        success = false;
        return NULL;
      }
      if (!(getOperatorType(context[1]) & operatorType::parenthesesStart)) {
        context.printError("Expected parenetheses with function arguments");
        success = false;
        return NULL;
      }

      function_t &func = *(new function_t(returnType,
                                          context[0]->to<identifierToken>()));
      context.pushPairRange(1);
      setArguments(func);
      context.popAndSkip();

      const opType_t opType = getOperatorType(context[0]);
      if (!(opType & (operatorType::semicolon |
                      operatorType::braceStart))) {
        context.printError("Expected a [;]");
        success = false;
        delete &func;
        return NULL;
      }

      // Copy attributes to the function itself
      func.attributes = attributes;
      // Make sure all attributes are meant for functions
      attributeTokenMap::iterator it = attributes.begin();
      while (it != attributes.end()) {
        attributeToken_t &attr = it->second;
        if (!attr.forFunction()) {
          attr.printError("Cannot apply attribute to function");
          success = false;
        }
        ++it;
      }

      // func(); <-- function
      if (opType & operatorType::semicolon) {
        context.set(1);
        return new functionStatement(up, func);
      }
      // func() {...} <-- function declaration
      functionDeclStatement &funcSmnt = *(new functionDeclStatement(up, func));
      success = funcSmnt.updateScope();
      if (!success) {
        success = false;
        delete &funcSmnt;
        // func wasn't added to scope, free it manually
        delete &func;
        return NULL;
      }
      addAttributesTo(attributes, &funcSmnt);

      pushUp(funcSmnt);
      statement_t *content = getNextStatement();
      popUp();
      if (success) {
        funcSmnt.set(*content);
      }
      if (!success) {
        delete &funcSmnt;
        return NULL;
      }
      return &funcSmnt;
    }

    void parser_t::checkIfConditionStatementExists() {
      // Called when checking:
      //     if (...), for (...), while (...), switch (...)
      // when the context is at
      //     if, for, while, switch

      // Need to make sure we have another token (even if it's not '(')
      bool error = (context.size() == 1);
      if (!error) {
        context.set(1);
        error = !(getOperatorType(context[0]) & operatorType::parenthesesStart);
      }

      if (error) {
        context.printError("Expected a condition statement");
        success = false;
      }
    }

    void parser_t::loadConditionStatements(statementPtrVector &statements,
                                           const int expectedCount) {
      // Load expression/declaration
      token_t *parenBegin = context[0];
      context.pushPairRange(0);

      int count = 0;
      bool error = true;
      while (true) {
        const int sType = peek();
        if (!success) {
          error = true;
          break;
        }
        if (success &&
            (sType & statementType::none)) {
          error = false;
          break;
        }

        if (count &&
            !(sType & (statementType::empty |
                       statementType::expression))) {
          parenBegin->printError("Expected an empty or expression statement");
          error = true;
          break;
        } else if (!count &&
                   !(sType & (statementType::empty      |
                              statementType::expression |
                              statementType::declaration))) {
          parenBegin->printError("Expected an empty, expression, or declaration statement");
          error = true;
          break;
        }

        ++count;
        if (count > expectedCount) {
          std::string message = "Too many statements, expected ";
          message += ('0' + (char) expectedCount);
          context.printError(message);
          error = true;
          break;
        }

        checkSemicolon = (count < expectedCount);
        statement_t *smnt = getNextStatement();
        statements.push_back(smnt);
        if (!success) {
          error = true;
          break;
        }
      }
      context.popAndSkip();

      if (!error &&
          (peek() & statementType::attribute)) {
        // TODO: Support multi-location errors and point to
        //         parenEnd as a suggestion
        context.printError("Attributes should be placed as an additional statement"
                           " (e.g. [for (;;; @attr)] or [if (; @attr)])");
        error = true;
      }

      const int smntCount = (int) statements.size();
      if (!success || error) {
        success = false;
        for (int i = 0; i < smntCount; ++i) {
          delete statements[i];
        }
        statements.clear();
        return;
      }
      if (smntCount) {
        statement_t *lastStatement = statements[smntCount - 1];
        if (lastStatement->type() & statementType::expression) {
          lastStatement->to<expressionStatement>().hasSemicolon = false;
        }
      }
    }

    statement_t* parser_t::loadConditionStatement() {
      statementPtrVector statements;
      loadConditionStatements(statements, 1);
      if (!statements.size()) {
        return NULL;
      }
      return statements[0];
    }

    statement_t* parser_t::loadIfStatement() {
      token_t *ifToken = context[0];
      checkIfConditionStatementExists();
      if (!success) {
        return NULL;
      }

      ifStatement &ifSmnt = *(new ifStatement(up, ifToken));
      pushUp(ifSmnt);

      statement_t *condition = loadConditionStatement();
      if (!condition) {
        if (success) {
          success = false;
          context.printError("Missing condition for [if] statement");
        }
        popUp();
        delete &ifSmnt;
        return NULL;
      }

      ifSmnt.setCondition(condition);
      addAttributesTo(attributes, &ifSmnt);

      statement_t *content = getNextStatement();
      if (!content) {
        if (success) {
          context.printError("Missing content for [if] statement");
          success = false;
        }
        delete &ifSmnt;
        popUp();
        return NULL;
      }
      ifSmnt.set(*content);

      int sType;
      while ((sType = peek()) & (statementType::elif_ |
                                 statementType::else_)) {
        pushUp(ifSmnt);
        statement_t *elSmnt = getNextStatement();
        popUp();
        if (!elSmnt) {
          if (success) {
            break;
          }
          delete &ifSmnt;
          popUp();
          return NULL;
        }
        if (sType & statementType::elif_) {
          ifSmnt.addElif(elSmnt->to<elifStatement>());
        } else {
          ifSmnt.addElse(elSmnt->to<elseStatement>());
          break;
        }
      }

      popUp();
      return &ifSmnt;
    }

    statement_t* parser_t::loadElifStatement() {
      // Skip [else] since checkIfConditionStatementExists
      //   expects 1 token before the condition
      // This is basically the same code as loadIfStatement
      //   but with an elif class
      token_t *elifToken = context[0];
      context.set(1);
      checkIfConditionStatementExists();
      if (!success) {
        return NULL;
      }

      elifStatement &elifSmnt = *(new elifStatement(up, elifToken));
      pushUp(elifSmnt);

      statement_t *condition = loadConditionStatement();
      if (!condition) {
        if (success) {
          success = false;
          context.printError("Missing condition for [else if] statement");
        }
        popUp();
        return NULL;
      }

      elifSmnt.setCondition(condition);
      addAttributesTo(attributes, &elifSmnt);

      statement_t *content = getNextStatement();
      popUp();
      if (!content) {
        context.printError("Missing content for [else if] statement");
        success = false;
        popUp();
        delete &elifSmnt;
        return NULL;
      }

      elifSmnt.set(*content);
      popUp();
      return &elifSmnt;
    }

    statement_t* parser_t::loadElseStatement() {
      token_t *elseToken = context[0];
      context.set(1);

      elseStatement &elseSmnt = *(new elseStatement(up, elseToken));
      addAttributesTo(attributes, &elseSmnt);

      pushUp(elseSmnt);
      statement_t *content = getNextStatement();
      popUp();
      if (!content) {
        context.printError("Missing content for [else] statement");
        success = false;
        delete &elseSmnt;
        return NULL;
      }

      elseSmnt.set(*content);
      return &elseSmnt;
    }

    statement_t* parser_t::loadForStatement() {
      token_t *forToken = context[0];
      checkIfConditionStatementExists();
      if (!success) {
        return NULL;
      }

      forStatement &forSmnt = *(new forStatement(up, forToken));
      pushUp(forSmnt);

      token_t *parenEnd = context.getClosingPairToken(0);

      statementPtrVector statements;
      loadConditionStatements(statements, 3);
      if (!success) {
        popUp();
        delete &forSmnt;
        return NULL;
      }

      int count = (int) statements.size();
      // Last statement is optional
      if (count == 2) {
        ++count;
        statements.push_back(new emptyStatement(up, parenEnd));
      }
      if (count < 3) {
        std::string message;
        if (count == 0) {
          message = "Expected [for] init and check statements";
        } else {
          message = "Expected [for] check statement";
        }
        if (parenEnd) {
          parenEnd->printError(message);
        } else {
          context.printError(message);
        }
        for (int i = 0; i < count; ++i) {
          delete statements[i];
        }
        success = false;
        popUp();
        delete &forSmnt;
        return NULL;
      }

      forSmnt.setLoopStatements(statements[0],
                                statements[1],
                                statements[2]);
      addAttributesTo(attributes, &forSmnt);
      if (!success) {
        delete &forSmnt;
        return NULL;
      }

      statement_t *content = getNextStatement();
      popUp();
      if (!content) {
        if (success) {
          context.printError("Missing content for [for] statement");
          success = false;
        }
        delete &forSmnt;
        return NULL;
      }

      forSmnt.set(*content);
      return &forSmnt;
    }

    statement_t* parser_t::loadWhileStatement() {
      token_t *whileToken = context[0];
      if (getKeyword(context[0]).type() & keywordType::do_) {
        return loadDoWhileStatement();
      }

      checkIfConditionStatementExists();
      if (!success) {
        return NULL;
      }

      whileStatement &whileSmnt = *(new whileStatement(up, whileToken));
      pushUp(whileSmnt);

      statement_t *condition = loadConditionStatement();
      if (!condition) {
        if (success) {
          success = false;
          context.printError("Missing condition for [while] statement");
        }
        popUp();
        delete &whileSmnt;
        return NULL;
      }

      whileSmnt.setCondition(condition);
      addAttributesTo(attributes, &whileSmnt);

      statement_t *content = getNextStatement();
      popUp();
      if (!content) {
        context.printError("Missing content for [while] statement");
        success = false;
        delete &whileSmnt;
        return NULL;
      }

      whileSmnt.set(*content);
      return &whileSmnt;
    }

    statement_t* parser_t::loadDoWhileStatement() {
      token_t *doToken = context[0];
      context.set(1);

      whileStatement &whileSmnt = *(new whileStatement(up, doToken, true));
      addAttributesTo(attributes, &whileSmnt);
      pushUp(whileSmnt);

      statement_t *content = getNextStatement();
      if (!content) {
        if (success) {
          context.printError("Missing content for [do-while] statement");
          success = false;
        }
        popUp();
        delete &whileSmnt;
        return NULL;
      }
      whileSmnt.set(*content);

      keyword_t &nextKeyword = getKeyword(context[0]);
      if (!(nextKeyword.type() & keywordType::while_)) {
        context.printError("Expected [while] condition after [do]");
        success = false;
        delete &whileSmnt;
        return NULL;
      }

      checkIfConditionStatementExists();
      if (!success) {
        delete &whileSmnt;
        popUp();
        return NULL;
      }

      statement_t *condition = loadConditionStatement();
      if (!condition) {
        if (success) {
          success = false;
          context.printError("Missing condition for [do-while] statement");
        }
        delete &whileSmnt;
        popUp();
        return NULL;
      }
      whileSmnt.setCondition(condition);

      if (!(getOperatorType(context[0]) & operatorType::semicolon)) {
        context.printError("Expected a [;]");
        success = false;
        popUp();
        delete &whileSmnt;
        return NULL;
      }
      context.set(1);

      return &whileSmnt;
    }

    statement_t* parser_t::loadSwitchStatement() {
      token_t *switchToken = context[0];
      checkIfConditionStatementExists();
      if (!success) {
        return NULL;
      }

      switchStatement &switchSmnt = *(new switchStatement(up, switchToken));
      pushUp(switchSmnt);

      token_t *parenEnd = context.getClosingPairToken(0);
      statement_t *condition = loadConditionStatement();
      if (!condition) {
        if (success) {
          success = false;
          context.printError("Missing condition for [switch] statement");
        }
        popUp();
        delete &switchSmnt;
        return NULL;
      }

      switchSmnt.setCondition(condition);
      addAttributesTo(attributes, &switchSmnt);

      statement_t *content = getNextStatement();
      popUp();
      if (!content) {
        parenEnd->printError("Missing content for [switch] statement");
        success = false;
        delete &switchSmnt;
        return NULL;
      }

      if (!(content->type() & (statementType::case_ |
                               statementType::default_))) {
        switchSmnt.set(*content);
      } else {
        switchSmnt.add(*content);

        content = getNextStatement();
        if (!content) {
          parenEnd->printError("Missing statement for switch's [case]");
          success = false;
          delete &switchSmnt;
          return NULL;
        }
        switchSmnt.add(*content);
      }

      return &switchSmnt;
    }

    statement_t* parser_t::loadCaseStatement() {
      token_t *caseToken = context[0];
      context.set(1);

      const int pos = context.getNextOperator(operatorType::colon);
      // No : found
      if (pos < 0) {
        context.printError("Expected a [:] to close the [case] statement");
        success = false;
        return NULL;
      }
      exprNode *value = NULL;
      // The case where we see 'case:'
      if (0 < pos) {
        // Load the case expression
        value = getExpression(0, pos);
      }
      if (!value) {
        context.printError("Expected a constant expression for the [case] statement");
        success = false;
        return NULL;
      }

      context.set(pos + 1);
      return new caseStatement(up, caseToken, *value);
    }

    statement_t* parser_t::loadDefaultStatement() {
      token_t *defaultToken = context[0];
      context.set(1);
      if (!(getOperatorType(context[0]) & operatorType::colon)) {
        context.printError("Expected a [:]");
        success = false;
        return NULL;
      }
      context.set(1);
      return new defaultStatement(up, defaultToken);
    }

    statement_t* parser_t::loadContinueStatement() {
      token_t *continueToken = context[0];
      context.set(1);
      if (!(getOperatorType(context[0]) & operatorType::semicolon)) {
        context.printError("Expected a [;]");
        success = false;
        return NULL;
      }
      context.set(1);
      return new continueStatement(up, continueToken);
    }

    statement_t* parser_t::loadBreakStatement() {
      token_t *breakToken = context[0];
      context.set(1);
      if (!(getOperatorType(context[0]) & operatorType::semicolon)) {
        context.printError("Expected a [;]");
        success = false;
        return NULL;
      }
      context.set(1);
      return new breakStatement(up, breakToken);
    }

    statement_t* parser_t::loadReturnStatement() {
      token_t *returnToken = context[0];
      context.set(1);

      const int pos = context.getNextOperator(operatorType::semicolon);
      // No ; found
      if (pos < 0) {
        context.printErrorAtEnd("Expected a [;]");
        success = false;
        return NULL;
      }
      exprNode *value = NULL;
      // The case where we see 'return;'
      if (0 < pos) {
        // Load the return value
        value = getExpression(0, pos);
      }
      if (!success) {
        return NULL;
      }

      context.set(pos + 1);
      return new returnStatement(up, returnToken, value);
    }

    statement_t* parser_t::loadClassAccessStatement() {
      token_t *accessToken = context[0];
      if (!(getOperatorType(context[1]) & operatorType::colon)) {
        context.printError("Expected a [:]");
        success = false;
        return NULL;
      }
      const int kType = getKeyword(context[0]).type();
      context.set(2);

      int access = classAccess::private_;
      if (kType == keywordType::public_) {
        access = classAccess::public_;
      } else if (kType == keywordType::protected_) {
        access = classAccess::protected_;
      }

      return new classAccessStatement(up, accessToken, access);
    }

    statement_t* parser_t::loadPragmaStatement() {
      pragmaStatement *smnt = new pragmaStatement(up,
                                                  context[0]
                                                  ->clone()
                                                  ->to<pragmaToken>());
      context.set(1);
      return smnt;
    }

    statement_t* parser_t::loadGotoStatement() {
      context.set(1);
      if (!(token_t::safeType(context[0]) & tokenType::identifier)) {
        context.printError("Expected [goto label] identifier");
        success = false;
        return NULL;
      }
      if (!(getOperatorType(context[1]) & operatorType::semicolon)) {
        context.printError("Expected a [;]");
        success = false;
        return NULL;
      }

      identifierToken &labelToken = (context[0]
                                     ->clone()
                                     ->to<identifierToken>());
      context.set(2);

      return new gotoStatement(up, labelToken);
    }

    statement_t* parser_t::loadGotoLabelStatement() {
      if (!(getOperatorType(context[1]) & operatorType::colon)) {
        context.printError("Expected a [:]");
        success = false;
        return NULL;
      }

      identifierToken &labelToken = (context[0]
                                     ->clone()
                                     ->to<identifierToken>());
      context.set(2);

      return new gotoLabelStatement(up, labelToken);
    }
    //==================================
  }
}
