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
#include <occa/io.hpp>
#include <occa/lang/attribute.hpp>
#include <occa/lang/expression.hpp>
#include <occa/lang/parser.hpp>
#include <occa/lang/variable.hpp>
#include <occa/lang/builtins/attributes.hpp>
#include <occa/lang/builtins/transforms.hpp>
#include <occa/lang/builtins/types.hpp>

namespace occa {
  namespace lang {
    parser_t::parser_t(const occa::properties &settings_) :
      preprocessor(settings_),
      unknownFilter(true),
      lastPeek(0),
      lastPeekPosition(-1),
      checkSemicolon(true),
      defaultRootToken(originSource::builtin),
      root(NULL, NULL),
      up(&root),
      success(true),
      settings(settings_) {
      // Properly implement `identifier-nondigit` for identifiers
      // Meanwhile, we use the unknownFilter
      stream = (tokenizer
                .filter(unknownFilter)
                .map(preprocessor)
                .map(stringMerger)
                .map(externMerger)
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

      getKeywords(keywords);

      addAttribute<attributes::dim>();
      addAttribute<attributes::dimOrder>();
      addAttribute<attributes::tile>();
    }

    parser_t::~parser_t() {
      clear();

      freeKeywords(keywords);

      nameToAttributeMap::iterator it = attributeMap.begin();
      while (it != attributeMap.end()) {
        delete it->second;
        ++it;
      }
      attributeMap.clear();
    }

    //---[ Public ]---------------------
    bool parser_t::succeeded() const {
      return success;
    }

    std::string parser_t::toString() const {
      return root.toString();
    }

    void parser_t::toString(std::string &s) const {
      s = root.toString();
    }

    void parser_t::writeToFile(const std::string &filename) const {
      io::write(filename,
                root.toString());
    }

    void parser_t::setMetadata(kernelMetadataMap &metadataMap) const {
      statementPtrVector kernelSmnts;
      findStatementsByAttr(statementType::functionDecl,
                           "kernel",
                           const_cast<blockStatement&>(root),
                           kernelSmnts);

      const int kernelCount = (int) kernelSmnts.size();
      for (int i = 0; i < kernelCount; ++i) {
        functionDeclStatement &declSmnt = *((functionDeclStatement*) kernelSmnts[i]);
        function_t &func = declSmnt.function;

        kernelMetadata &metadata = metadataMap[func.name()];
        metadata.name = func.name();

        int args = (int) func.args.size();
        for (int ai = 0; ai < args; ++ai) {
          metadata += argumentInfo(
            func.args[ai]->has(const_)
          );
        }
      }
    }
    //==================================

    //---[ Setup ]----------------------
    void parser_t::clear() {
      tokenizer.clear();
      context.clear();

      preprocessor.clear();
      addSettingDefines();

      lastPeek = 0;
      lastPeekPosition = -1;
      checkSemicolon = true;

      root.clear();
      delete root.source;
      root.source = NULL;

      up = &root;
      upStack.clear();

      clearAttributes();

      onClear();

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

    void parser_t::addSettingDefines() {
      json &defines = settings["defines"];
      if (defines.isObject()) {
        jsonObject &defineMap = defines.object();
        jsonObject::iterator it = defineMap.begin();
        while (it != defineMap.end()) {
          const std::string &define = it->first;
          json &value = it->second;

          preprocessor.addSourceDefine(define, value);

          ++it;
        }
      }
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

      delete root.source;
      root.source = (
        context.size()
        ? context[0]->clone()
        : defaultRootToken.clone()
      );
    }

    void parser_t::loadTokens() {
      beforePreprocessing();

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
    }

    void parser_t::parseTokens() {
      beforeParsing();
      if (!success) return;
      loadAllStatements();
      if (!success) return;
      success = transforms::applyDimTransforms(root);
      if (!success) return;
      success = transforms::applyTileTransforms(root);
      if (!success) return;
      afterParsing();
    }

    keyword_t& parser_t::getKeyword(token_t *token) {
      static keyword_t noKeyword;
      if (!token) {
        return noKeyword;
      }

      const int tType = token->type();
      if (!(tType & (tokenType::identifier |
                     tokenType::qualifier  |
                     tokenType::type       |
                     tokenType::variable   |
                     tokenType::function))) {
        return noKeyword;
      }

      std::string identifier;
      if (tType & tokenType::identifier) {
        identifier = token->to<identifierToken>().value;
      }
      else if (tType & tokenType::qualifier) {
        identifier = token->to<qualifierToken>().qualifier.name;
      }
      else if (tType & tokenType::type) {
        identifier = token->to<typeToken>().value.name();
      }
      else if (tType & tokenType::variable) {
        identifier = token->to<variableToken>().value.name();
      }
      else if (tType & tokenType::function) {
        identifier = token->to<functionToken>().value.name();
      }

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
      context.push(start, end);
      const int tokenCount = context.size();
      tokenVector tokens;
      tokens.reserve(tokenCount);

      if (up) {
        // Replace identifier tokens with keywords if they exist
        for (int i = 0; i < tokenCount; ++i) {
          token_t *token = context[i];
          if (token->type() & tokenType::identifier) {
            context.setToken(i, replaceIdentifier((identifierToken&) *token));
          }
        }
        while (context.size()) {
          token_t *token = context[0];
          if (!(token->type() & (tokenType::qualifier |
                                 tokenType::type))) {
            context.set(1);
            tokens.push_back(token->clone());
            continue;
          }

          vartype_t vartype = loadType();
          if (!success) {
            context.pop();
            freeTokenVector(tokens);
            return NULL;
          }

          tokens.push_back(new vartypeToken(token->origin,
                                            vartype));
        }
      }
      context.pop();

      exprNode *expr = occa::lang::getExpression(tokens);
      success &= !!expr;
      return expr;
    }

    token_t* parser_t::replaceIdentifier(identifierToken &identifier) {
      keyword_t &keyword = getKeyword(&identifier);
      const int kType = keyword.type();

      if (!(kType & (keywordType::qualifier |
                     keywordType::type      |
                     keywordType::variable  |
                     keywordType::function))) {
        return &identifier;
      }

      if (kType & keywordType::qualifier) {
        return new qualifierToken(identifier.origin,
                                  ((qualifierKeyword&) keyword).qualifier);
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

    attribute_t* parser_t::getAttribute(const std::string &name) {
      nameToAttributeMap::iterator it = attributeMap.find(name);
      if (it == attributeMap.end()) {
        return NULL;
      }
      return it->second;
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
        context.printError("Expected an attribute name");
        success = false;
        return;
      }

      identifierToken &nameToken = *((identifierToken*) context[0]);
      context.set(1);

      attribute_t *attribute = getAttribute(nameToken.value);
      if (!attribute) {
        nameToken.printError("Unknown attribute");
        success = false;
        return;
      }

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

      attributeToken_t attr(*attribute, nameToken);
      setAttributeArgs(attr, argRanges);
      if (hasArgs) {
        context.popAndSkip();
      }
      if (success) {
        attrs[nameToken.value] = attr;
        success = attribute->isValid(attr);
      }
    }

    void parser_t::setAttributeArgs(attributeToken_t &attr,
                                    tokenRangeVector &argRanges) {
      const int args = (int) argRanges.size();
      bool foundNamedArg = false;
      for (int i = 0; i < args; ++i) {
        attributeArg_t arg;
        context.push(argRanges[i].start,
                     argRanges[i].end);

        if (!context.size()) {
          arg.expr = new emptyNode();
          attr.args.push_back(arg);
          context.popAndSkip();
          continue;
        }

        // Load args
        loadAttributes(arg.attributes);
        if (!context.size()) {
          arg.expr = new emptyNode();
          attr.args.push_back(arg);
          context.popAndSkip();
          continue;
        }

        // Get argument
        arg.expr = getExpression();
        if (!success) {
          context.pop();
          arg.clear();
          return;
        }

        std::string argName;
        // Check for
        // |---[=] (binary)
        // |   |
        // |   |---[argName] (identifier)
        // |   |
        // |   |---[arg] (?)
        if (arg.expr->type() & exprNodeType::binary) {
          binaryOpNode &equalsNode = arg.expr->to<binaryOpNode>();
          if ((equalsNode.opType() & operatorType::assign) &&
              (equalsNode.leftValue->type() & exprNodeType::identifier)) {
            argName = equalsNode.leftValue->to<identifierNode>().value;
            arg.expr = equalsNode.rightValue->clone();
            delete &equalsNode;
          }
        }

        if (!argName.size() &&
            foundNamedArg) {
          context.printError("All arguments after a named argument"
                             " must also be named");
          success = false;
          context.pop();
          arg.clear();
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
          smnt->printError("Trying to add @" + attr.name() + " to this statement");
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

      vartype_t vartype = loadType();
      variable_t var = (!isLoadingFunctionPointer()
                        ? loadVariable(vartype)
                        : loadFunctionPointer(vartype));

      if (var.vartype.type) {
        var.attributes = var.vartype.type->attributes;
      }
      var.attributes.insert(attrs.begin(), attrs.end());

      return var;
    }

    variableDeclaration parser_t::loadVariableDeclaration(attributeTokenMap &smntAttributes,
                                                          const vartype_t &baseType) {
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

      loadDeclarationAttributes(smntAttributes, decl);
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


    void parser_t::loadDeclarationAttributes(attributeTokenMap &smntAttributes,
                                             variableDeclaration &decl) {
      attributeTokenMap &varAttributes = decl.variable->attributes;
      // Copy statement attributes to each variable
      // Variable attributes should override statement attributes
      varAttributes.insert(smntAttributes.begin(),
                           smntAttributes.end());
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
          context.printErrorAtEnd("[1] Expected a [;]");
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
      if (!success) {
        return;
      }
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
      if (!success) {
        return;
      }
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

    vartype_t parser_t::loadType() {
      // TODO: Handle weird () cases:
      //        int (*const (*const a))      -> int * const * const a;
      //        int (*const (*const (*a)))() -> int (* const * const *a)();
      // Set the name in loadBaseType and look for (*)() or (^)()
      //   to stop qualifier merging
      vartype_t vartype;
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
          vartype.typeToken = (identifierToken*) token->clone();
          continue;
        }
        break;
      }

      if (tokenPos == 0) {
        context.printError("Unable to load type");
        success = false;
        return;
      }

      context.set(tokenPos);

      if (vartype.isValid()) {
        return;
      }

      if (vartype.has(long_) ||
          vartype.has(longlong_)) {
        vartype.type = &int_;
        return;
      }

      context.printError("Expected a type");
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

      vartype_t vartype = loadType();
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

        exprNode *value = NULL;
        if (context.size()) {
          value = getExpression();
          if (!success) {
            return;
          }
        }
        vartype += array_t(start,
                           end,
                           value);

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
        // Copy attributes before continuing to avoid passing them to
        //   nested statements
        attributeTokenMap smntAttributes = attributes;
        attributes.clear();

        statementLoader_t loader = it->second;
        statement_t *smnt = (this->*loader)(smntAttributes);
        if (!smnt) {
          return NULL;
        }
        if (!success) {
          delete smnt;
          return NULL;
        }

        // [checkSemicolon] is only valid for one statement
        checkSemicolon = true;
        addAttributesTo(smntAttributes, smnt);
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

    statement_t* parser_t::loadBlockStatement(attributeTokenMap &smntAttributes) {
      blockStatement *smnt = new blockStatement(up, context[0]);
      addAttributesTo(smntAttributes, smnt);

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

    statement_t* parser_t::loadEmptyStatement(attributeTokenMap &smntAttributes) {
      statement_t *smnt = new emptyStatement(up, context[0]);
      addAttributesTo(smntAttributes, smnt);

      // Skip [;] token
      context.set(1);
      return smnt;
    }

    statement_t* parser_t::loadExpressionStatement(attributeTokenMap &smntAttributes) {
      int end = context.getNextOperator(operatorType::semicolon);
      if (end < 0) {
        if (checkSemicolon) {
          context.printErrorAtEnd("[2] Expected a [;]");
          success = false;
          return NULL;
        }
        end = context.size();
      }

      context.push(0, end);
      exprNode *expr = getExpression();
      context.pop();
      if (!success) {
        return NULL;
      }
      context.set(end + 1);

      expressionStatement *smnt = new expressionStatement(up, *expr);
      addAttributesTo(smntAttributes, smnt);
      return smnt;
    }

    statement_t* parser_t::loadDeclarationStatement(attributeTokenMap &smntAttributes) {
      bool isFunction = isLoadingFunction();
      if (!success) {
        return NULL;
      }
      if (isFunction) {
        return loadFunctionStatement(smntAttributes);
      }

      vartype_t baseType;
      loadBaseType(baseType);
      if (!success) {
        return NULL;
      }

      declarationStatement &smnt = *(new declarationStatement(up));
      addAttributesTo(smntAttributes, &smnt);
      while(success) {
        // TODO: Pass decl as argument to prevent a copy
        success = smnt.addDeclaration(
          loadVariableDeclaration(smnt.attributes, baseType)
        );
        if (!success) {
          break;
        }
        const opType_t opType = getOperatorType(context[0]);
        if (!(opType & operatorType::comma)) {
          if (opType & operatorType::semicolon) {
            context.set(1);
          } else if (checkSemicolon) {
            context.printError("[3] Expected a [;]");
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

    statement_t* parser_t::loadNamespaceStatement(attributeTokenMap &smntAttributes) {
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
      // Add attributes to the most-nested namespace
      // TODO: Should this be the attribute logic?
      addAttributesTo(smntAttributes, currentSmnt);

      // Load block content
      context.pushPairRange(0);
      pushUp(*currentSmnt);
      loadAllStatements();
      popUp();
      context.popAndSkip();

      return smnt;
    }

    statement_t* parser_t::loadTypeDeclStatement(attributeTokenMap &smntAttributes) {
      return NULL;
    }

    statement_t* parser_t::loadFunctionStatement(attributeTokenMap &smntAttributes) {
      vartype_t returnType = loadType();

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
        context.printError("[4] Expected a [;]");
        success = false;
        delete &func;
        return NULL;
      }

      // Copy attributes to the function itself
      func.attributes = smntAttributes;
      // Make sure all attributes are meant for functions
      attributeTokenMap::iterator it = smntAttributes.begin();
      while (it != smntAttributes.end()) {
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
      addAttributesTo(smntAttributes, &funcSmnt);

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

      if (smntCount &&
          (smntCount == expectedCount)) {
        statement_t *lastStatement = statements[smntCount - 1];
        const int lastType = lastStatement->type();
        if (lastType & statementType::expression) {
          lastStatement->to<expressionStatement>().hasSemicolon = false;
        }
        else if (lastType & statementType::empty) {
          lastStatement->to<emptyStatement>().hasSemicolon = false;
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

    statement_t* parser_t::loadIfStatement(attributeTokenMap &smntAttributes) {
      token_t *ifToken = context[0];
      checkIfConditionStatementExists();
      if (!success) {
        return NULL;
      }

      ifStatement &ifSmnt = *(new ifStatement(up, ifToken));
      pushUp(ifSmnt);
      addAttributesTo(smntAttributes, &ifSmnt);

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

    statement_t* parser_t::loadElifStatement(attributeTokenMap &smntAttributes) {
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
      addAttributesTo(smntAttributes, &elifSmnt);

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

    statement_t* parser_t::loadElseStatement(attributeTokenMap &smntAttributes) {
      token_t *elseToken = context[0];
      context.set(1);

      elseStatement &elseSmnt = *(new elseStatement(up, elseToken));
      pushUp(elseSmnt);
      addAttributesTo(smntAttributes, &elseSmnt);

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

    statement_t* parser_t::loadForStatement(attributeTokenMap &smntAttributes) {
      token_t *forToken = context[0];
      checkIfConditionStatementExists();
      if (!success) {
        return NULL;
      }

      forStatement &forSmnt = *(new forStatement(up, forToken));
      pushUp(forSmnt);
      addAttributesTo(smntAttributes, &forSmnt);

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
        statements.push_back(new emptyStatement(up, parenEnd, false));
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
      if (!success) {
        delete &forSmnt;
        return NULL;
      }
      // If the last statement had attributes, we need to pass them now
      addAttributesTo(attributes, &forSmnt);

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

    statement_t* parser_t::loadWhileStatement(attributeTokenMap &smntAttributes) {
      token_t *whileToken = context[0];
      if (getKeyword(context[0]).type() & keywordType::do_) {
        return loadDoWhileStatement(smntAttributes);
      }

      checkIfConditionStatementExists();
      if (!success) {
        return NULL;
      }

      whileStatement &whileSmnt = *(new whileStatement(up, whileToken));
      pushUp(whileSmnt);
      addAttributesTo(smntAttributes, &whileSmnt);

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

    statement_t* parser_t::loadDoWhileStatement(attributeTokenMap &smntAttributes) {
      token_t *doToken = context[0];
      context.set(1);

      whileStatement &whileSmnt = *(new whileStatement(up, doToken, true));
      pushUp(whileSmnt);
      addAttributesTo(smntAttributes, &whileSmnt);

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
        context.printError("[5] Expected a [;]");
        success = false;
        popUp();
        delete &whileSmnt;
        return NULL;
      }
      context.set(1);

      return &whileSmnt;
    }

    statement_t* parser_t::loadSwitchStatement(attributeTokenMap &smntAttributes) {
      token_t *switchToken = context[0];
      checkIfConditionStatementExists();
      if (!success) {
        return NULL;
      }

      switchStatement &switchSmnt = *(new switchStatement(up, switchToken));
      pushUp(switchSmnt);
      addAttributesTo(smntAttributes, &switchSmnt);

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

    statement_t* parser_t::loadCaseStatement(attributeTokenMap &smntAttributes) {
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

      caseStatement *smnt = new caseStatement(up, caseToken, *value);
      addAttributesTo(smntAttributes, smnt);
      return smnt;
    }

    statement_t* parser_t::loadDefaultStatement(attributeTokenMap &smntAttributes) {
      token_t *defaultToken = context[0];
      context.set(1);
      if (!(getOperatorType(context[0]) & operatorType::colon)) {
        context.printError("Expected a [:]");
        success = false;
        return NULL;
      }
      context.set(1);

      defaultStatement *smnt = new defaultStatement(up, defaultToken);
      addAttributesTo(smntAttributes, smnt);
      return smnt;
    }

    statement_t* parser_t::loadContinueStatement(attributeTokenMap &smntAttributes) {
      token_t *continueToken = context[0];
      context.set(1);
      if (!(getOperatorType(context[0]) & operatorType::semicolon)) {
        context.printError("[6] Expected a [;]");
        success = false;
        return NULL;
      }
      context.set(1);

      continueStatement *smnt = new continueStatement(up, continueToken);
      addAttributesTo(smntAttributes, smnt);
      return smnt;
    }

    statement_t* parser_t::loadBreakStatement(attributeTokenMap &smntAttributes) {
      token_t *breakToken = context[0];
      context.set(1);
      if (!(getOperatorType(context[0]) & operatorType::semicolon)) {
        context.printError("[7] Expected a [;]");
        success = false;
        return NULL;
      }
      context.set(1);

      breakStatement *smnt = new breakStatement(up, breakToken);
      addAttributesTo(smntAttributes, smnt);
      return smnt;
    }

    statement_t* parser_t::loadReturnStatement(attributeTokenMap &smntAttributes) {
      token_t *returnToken = context[0];
      context.set(1);

      const int pos = context.getNextOperator(operatorType::semicolon);
      // No ; found
      if (pos < 0) {
        context.printErrorAtEnd("[8] Expected a [;]");
        success = false;
        return NULL;
      }
      exprNode *value = NULL;
      // The case where we see 'return;'
      if (0 < pos) {
        // Load the return value
        value = getExpression(0, pos);
        if (!success) {
          return NULL;
        }
      }

      context.set(pos + 1);

      returnStatement *smnt = new returnStatement(up, returnToken, value);
      addAttributesTo(smntAttributes, smnt);
      return smnt;
    }

    statement_t* parser_t::loadClassAccessStatement(attributeTokenMap &smntAttributes) {
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

      classAccessStatement *smnt = new classAccessStatement(up, accessToken, access);
      addAttributesTo(smntAttributes, smnt);
      return smnt;
    }

    statement_t* parser_t::loadPragmaStatement(attributeTokenMap &smntAttributes) {
      pragmaStatement *smnt = new pragmaStatement(up,
                                                  *((pragmaToken*) context[0]));
      addAttributesTo(smntAttributes, smnt);

      context.set(1);

      return smnt;
    }

    statement_t* parser_t::loadGotoStatement(attributeTokenMap &smntAttributes) {
      context.set(1);
      if (!(token_t::safeType(context[0]) & tokenType::identifier)) {
        context.printError("Expected [goto label] identifier");
        success = false;
        return NULL;
      }
      if (!(getOperatorType(context[1]) & operatorType::semicolon)) {
        context.printError("[9] Expected a [;]");
        success = false;
        return NULL;
      }

      identifierToken &labelToken = (context[0]
                                     ->clone()
                                     ->to<identifierToken>());
      context.set(2);

      gotoStatement *smnt = new gotoStatement(up, labelToken);
      addAttributesTo(smntAttributes, smnt);
      return smnt;
    }

    statement_t* parser_t::loadGotoLabelStatement(attributeTokenMap &smntAttributes) {
      if (!(getOperatorType(context[1]) & operatorType::colon)) {
        context.printError("Expected a [:]");
        success = false;
        return NULL;
      }

      identifierToken &labelToken = (context[0]
                                     ->clone()
                                     ->to<identifierToken>());
      context.set(2);

      gotoLabelStatement *smnt = new gotoLabelStatement(up, labelToken);
      addAttributesTo(smntAttributes, smnt);
      return smnt;
    }
    //==================================

    //---[ Customization ]--------------
    void parser_t::onClear() {}
    void parser_t::beforePreprocessing() {}
    void parser_t::beforeParsing() {}
    void parser_t::afterParsing() {}
    //==================================
  }
}
