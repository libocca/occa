#include <occa/internal/io.hpp>
#include <occa/internal/lang/attribute.hpp>
#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/parser.hpp>
#include <occa/internal/lang/variable.hpp>
#include <occa/internal/lang/builtins/attributes.hpp>
#include <occa/internal/lang/builtins/types.hpp>
#include <occa/utils/hash.hpp>

namespace occa {
  namespace lang {
    parser_t::parser_t(const occa::json &settings_) :
      preprocessor(settings_),
      unknownFilter(true),
      root(NULL, NULL),
      smntContext(root),
      smntPeeker(tokenContext,
                 smntContext,
                 *this,
                 attributeMap),
      loadingStatementType(0),
      checkSemicolon(true),
      defaultRootToken(originSource::builtin),
      success(true),
      settings(settings_),
      restrictQualifier(NULL) {
      // Properly implement `identifier-nondigit` for identifiers
      // Meanwhile, we use the unknownFilter
      stream = (tokenizer
                .filter(unknownFilter)
                .map(preprocessor)
                .map(stringMerger)
                .map(externMerger)
                .map(newlineFilter));

      // Statement type -> loader function
      statementLoaders[statementType::empty]       = &parser_t::loadEmptyStatement;
      statementLoaders[statementType::expression]  = &parser_t::loadExpressionStatement;
      statementLoaders[statementType::declaration] = &parser_t::loadDeclarationStatement;
      statementLoaders[statementType::block]       = &parser_t::loadBlockStatement;
      statementLoaders[statementType::namespace_]  = &parser_t::loadNamespaceStatement;
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
      statementLoaders[statementType::directive]   = &parser_t::loadDirectiveStatement;
      statementLoaders[statementType::pragma]      = &parser_t::loadPragmaStatement;
      statementLoaders[statementType::goto_]       = &parser_t::loadGotoStatement;
      statementLoaders[statementType::gotoLabel]   = &parser_t::loadGotoLabelStatement;

      getKeywords(keywords);

      addAttribute<attributes::atomic>();
      addAttribute<attributes::dim>();
      addAttribute<attributes::dimOrder>();
      addAttribute<attributes::tile>();
      addAttribute<attributes::occaRestrict>();
      addAttribute<attributes::implicitArg>();
      addAttribute<attributes::globalPtr>();
    }

    parser_t::~parser_t() {
      clear();

      keywords.free();
      delete restrictQualifier;

      for (auto it : attributeMap) {
        delete it.second;
      }
      attributeMap.clear();
    }

    //---[ Customization ]--------------
    void parser_t::onClear() {}
    void parser_t::beforePreprocessing() {}
    void parser_t::beforeParsing() {}
    void parser_t::afterParsing() {}
    //==================================

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

    void parser_t::setSourceMetadata(sourceMetadata_t &sourceMetadata) const {
      kernelMetadataMap &metadataMap = sourceMetadata.kernelsMetadata;
      strHashMap &dependencyHashes = sourceMetadata.dependencyHashes;

      // Set metadata for all @kernels
      root.children
        .forEachKernelStatement([&](functionDeclStatement &kernelSmnt) {
          function_t &func = kernelSmnt.function();

          kernelMetadata_t &metadata = metadataMap[func.name()];
          metadata.name = func.name();

          int args = (int) func.args.size();
          for (int ai = 0; ai < args; ++ai) {
            variable_t &arg = *(func.args[ai]);
            // Ignore implicit arguments that come from the device
            if (arg.hasAttribute("implicitArg")) {
              continue;
            }
            metadata += argMetadata_t(
              arg.has(const_),
              arg.vartype.isPointerType(),
              arg.dtype(),
              arg.name()
            );
          }
        });

      // Set dependencies and their hashes
      strVector dependencies = preprocessor.getDependencyFilenames();
      const int dependencyCount = (int) dependencies.size();
      for (int i = 0; i < dependencyCount; ++i) {
        const std::string &dependency = dependencies[i];
        dependencyHashes[dependency] = hashFile(dependency);
      }
    }
    //==================================

    //---[ Setup ]----------------------
    void parser_t::clear() {
      tokenizer.clear();

      root.clear();
      delete root.source;
      root.source = NULL;

      tokenContext.clear();
      smntContext.clear();
      smntPeeker.clear();

      preprocessor.clear();
      addSettingDefines();

      loadingStatementType = 0;
      checkSemicolon = true;

      comments.clear();
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
      if (!defines.isObject()) {
        return;
      }

      jsonObject &defineMap = defines.object();
      jsonObject::iterator it = defineMap.begin();
      while (it != defineMap.end()) {
        const std::string &define = it->first;
        json &value = it->second;

        preprocessor.addSourceDefine(define, value);

        ++it;
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
      stream.clearCache();

      if (isFile) {
        tokenizer.set(new file_t(source));
      } else {
        tokenizer.set(source.c_str());
      }

      setupLoadTokens();
      loadTokens();

      delete root.source;
      root.source = (
        tokenContext.size()
        ? tokenContext[0]->clone()
        : defaultRootToken.clone()
      );
    }

    void parser_t::setupLoadTokens() {
      beforePreprocessing();

      // Setup @restrict
      const std::string restrictStr = (
        settings.get<std::string>("okl/restrict",
                                  "__restrict__")
      );

      if (restrictStr != "disabled") {
        restrictQualifier = new qualifier_t(restrictStr,
                                            qualifierType::custom);
        keywords.add(*(new qualifierKeyword(*restrictQualifier)));
      }
    }

    void parser_t::loadTokens() {
      tokenVector tokens;
      token_t *token;
      while (!stream.isEmpty()) {
        stream >> token;
        tokens.push_back(token);
      }

      if (tokenizer.errors ||
          preprocessor.errors) {
        success = false;
        return;
      }

      tokenContext.setup(tokens);
      success &= !tokenContext.hasError;
    }

    void parser_t::parseTokens() {
      beforeParsing();
      if (!success) return;

      loadAllStatements();
      if (!success) return;

      if (restrictQualifier) {
        success &= attributes::occaRestrict::applyCodeTransformations(root, *restrictQualifier);
        if (!success) return;
      }

      success &= attributes::dim::applyCodeTransformations(root);
      if (!success) return;

      success &= attributes::tile::applyCodeTransformations(root);
      if (!success) return;

      afterParsing();
    }
    //==================================

    //---[ Helper Methods ]-------------
    keyword_t& parser_t::getKeyword(token_t *token) {
      return keywords.get(smntContext, token);
    }

    keyword_t& parser_t::getKeyword(const std::string &name) {
      return keywords.get(smntContext, name);
    }

    exprNode* parser_t::parseTokenContextExpression() {
      exprNode *expr = tokenContext.parseExpression(smntContext, *this);
      success &= !!expr;
      return expr;
    }

    exprNode* parser_t::parseTokenContextExpression(const int start,
                                                    const int end) {
      exprNode *expr = tokenContext.parseExpression(smntContext,
                                                    *this,
                                                    start, end);
      success &= !!expr;
      return expr;
    }

    void parser_t::loadComments() {
      const int start = tokenContext.position();
      loadComments(start, start);
    }

    void parser_t::loadComments(const int start,
                                const int end) {
      tokenVector skippedTokens;
      tokenContext.getSkippedTokens(skippedTokens, start, end);

      const int skippedTokenCount = (int) skippedTokens.size();
      if (!skippedTokenCount) {
        return;
      }

      for (int i = 0; i < skippedTokenCount; ++i) {
        token_t *token = skippedTokens[i];
        if (!(token->type() & tokenType::comment)) {
          continue;
        }

        comments.push(
          new commentStatement(smntContext.up,
                               *((commentToken*) token))
        );
      }

      // Push comments if we're in the root statement
      if (smntContext.up == &root) {
        pushComments();
      }
    }

    void parser_t::pushComments() {
      const int commentsCount = (int) comments.length();
      for (int i = 0; i < commentsCount; ++i) {
        statement_t *smnt = comments[i];
        smnt->up = smntContext.up;
        smntContext.up->children.push(smnt);
      }
      comments.clear();
    }

    void parser_t::loadAttributes(attributeTokenMap &attrs) {
      success &= lang::loadAttributes(tokenContext,
                                      smntContext,
                                      *this,
                                      attributeMap,
                                      attrs);
    }

    attribute_t* parser_t::getAttribute(const std::string &name) {
      return lang::getAttribute(attributeMap, name);
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
        if (attr.forStatementType(sType)) {
          smnt->addAttribute(attr);
          ++it;
          continue;
        }

        attr.printError("Cannot apply attribute to a ["
                        + smnt->statementName()
                        + "] statement");
        smnt->printError("Trying to add @" + attr.name() + " to this statement");
        smnt->attributes.clear();
        clearAttributes(attrs);
        success = false;
        break;
      }
      attrs.clear();
    }

    void parser_t::loadBaseType(vartype_t &vartype) {
      success &= lang::loadBaseType(tokenContext,
                                    smntContext,
                                    *this,
                                    vartype);
    }

    void parser_t::loadType(vartype_t &vartype) {
      success &= lang::loadType(tokenContext,
                                smntContext,
                                *this,
                                vartype);
    }

    vartype_t parser_t::loadType() {
      vartype_t vartype;
      success &= lang::loadType(tokenContext,
                                smntContext,
                                *this,
                                vartype);
      return vartype;
    }

    bool parser_t::isLoadingVariable() {
      return lang::isLoadingVariable(tokenContext,
                                     smntContext,
                                     *this,
                                     attributeMap);
    }

    bool parser_t::isLoadingFunction() {
      return lang::isLoadingFunction(tokenContext,
                                     smntContext,
                                     *this,
                                     attributeMap);
    }

    bool parser_t::isLoadingFunctionPointer() {
      return lang::isLoadingFunctionPointer(tokenContext,
                                            smntContext,
                                            *this,
                                            attributeMap);
    }

    void parser_t::loadVariable(variable_t &var) {
      success &= lang::loadVariable(tokenContext,
                                    smntContext,
                                    *this,
                                    attributeMap,
                                    var);
    }

    variable_t parser_t::loadVariable() {
      variable_t var;
      loadVariable(var);
      return var;
    }

    void parser_t::loadVariable(vartype_t &vartype,
                                variable_t &var) {
      success &= lang::loadVariable(tokenContext,
                                    smntContext,
                                    *this,
                                    attributeMap,
                                    vartype,
                                    var);
    }

    void parser_t::loadFunction(function_t &func) {
      success &= lang::loadFunction(tokenContext,
                                    smntContext,
                                    *this,
                                    attributeMap,
                                    func);
    }

    int parser_t::peek() {
      const int tokenContextStart = tokenContext.position();

      // Peek skips tokens when loading attributes
      int sType;
      success &= smntPeeker.peek(attributes,
                                 sType);

      const int tokenContextEnd = tokenContext.position();

      // Load comments between skipped tokens
      if (tokenContextStart != tokenContextEnd) {
        loadComments(tokenContextStart, tokenContextEnd);
      }

      return sType;
    }
    //==================================

    //---[ Type Loaders ]---------------
    variableDeclaration parser_t::loadVariableDeclaration(attributeTokenMap &smntAttributes,
                                                          const vartype_t &baseType) {
      variableDeclaration decl;

      // If partially-defined type, finish parsing it
      vartype_t vartype = baseType.declarationType();
      vartype.qualifiers = baseType.qualifiers;

      variable_t var;
      loadVariable(vartype, var);
      decl.setVariable(var.clone());

      applyDeclarationSmntAttributes(smntAttributes,
                                     decl.variable());

      loadDeclarationBitfield(decl);
      loadDeclarationAssignment(decl);
      if (!decl.value) {
        loadDeclarationBraceInitializer(decl);
      }

      return decl;
    }


    void parser_t::applyDeclarationSmntAttributes(attributeTokenMap &smntAttributes,
                                                  variable_t &var) {
      attributeTokenMap allAttributes = smntAttributes;
      attributeTokenMap &varAttributes = var.attributes;
      // Copy statement attributes to each variable
      // Variable attributes should override statement attributes
      allAttributes.insert(varAttributes.begin(),
                           varAttributes.end());
      varAttributes = allAttributes;

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
      int pos = tokenContext.getNextOperator(opCheck);
      if (pos < 0) {
        if (checkSemicolon) {
          tokenContext.printErrorAtEnd("[1] Expected a [;]");
          success = false;
        }
        pos = tokenContext.size();
      }
      return pos;
    }

    void parser_t::loadDeclarationBitfield(variableDeclaration &decl) {
      if (!(token_t::safeOperatorType(tokenContext[0]) & operatorType::colon)) {
        return;
      }

      int pos = declarationNextCheck(operatorType::assign     |
                                     operatorType::braceStart |
                                     operatorType::comma      |
                                     operatorType::semicolon);
      if (pos == 1) {
        tokenContext[1]->printError("Expected an expression");
        success = false;
      }
      if (!success) {
        return;
      }

      exprNode *value = parseTokenContextExpression(1, pos);
      if (!success) {
        return;
      }
      decl.variable().vartype.bitfield = (int) value->evaluate();
      delete value;
      tokenContext += pos;
    }

    void parser_t::loadDeclarationAssignment(variableDeclaration &decl) {
      if (!(token_t::safeOperatorType(tokenContext[0]) & operatorType::assign)) {
        return;
      }

      int pos = declarationNextCheck(operatorType::comma |
                                     operatorType::semicolon);
      if (pos == 1) {
        tokenContext[1]->printError("Expected an expression");
        success = false;
      }
      if (!success) {
        return;
      }

      decl.value = parseTokenContextExpression(1, pos);
      if (!success) {
        return;
      }
      tokenContext += pos;
    }

    // TODO: Store brace initializer propertly, maybe in value with an extra flag
    void parser_t::loadDeclarationBraceInitializer(variableDeclaration &decl) {
      if (!(token_t::safeOperatorType(tokenContext[0]) & operatorType::braceStart)) {
        return;
      }

      tokenContext.push(tokenContext.getClosingPair() + 1);
      int pos = declarationNextCheck(operatorType::comma |
                                     operatorType::semicolon);
      if ((pos != 0) &&
          (pos != tokenContext.size())) {
        tokenContext.printError("Expected a [,] for another variable"
                                " or a stopping [;]");
        success = false;
      }
      if (!success) {
        return;
      }
      tokenContext.pop();

      tokenContext.pushPairRange();
      decl.value = parseTokenContextExpression();
      tokenContext.popAndSkip();
    }
    //==================================

    //---[ Statement Loaders ]----------
    bool parser_t::isEmpty() {
      const int sType = peek();
      return (!success ||
              (sType & statementType::none));
    }

    void parser_t::loadAllStatements() {
      statementArray &statements = smntContext.up->children;
      statement_t *smnt = getNextStatement();

      while (smnt) {
        statements.push(smnt);
        smnt = getNextStatement();
      }

      // Load comments at the end of the block
      loadComments();
      pushComments();
    }

    statement_t* parser_t::loadNextStatement() {
      if (isEmpty()) {
        checkSemicolon = true;
        return NULL;
      }

      loadComments();

      const int sType = peek();
      if (!success) {
        return NULL;
      }

      if (sType & statementType::blockStatements) {
        pushComments();
      }

      statementLoaderMap::iterator it = statementLoaders.find(sType);
      if (it != statementLoaders.end()) {
        // Copy attributes before continuing to avoid passing them to
        //   nested statements
        attributeTokenMap smntAttributes = attributes;
        attributes.clear();

        loadingStatementType = sType;

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

    statement_t* parser_t::getNextStatement() {
      statement_t *smnt = loadNextStatement();

      // It's the end or we don't have comments
      if (!smnt || !comments.length()) {
        return smnt;
      }

      // We're about to load a block statement type, add the comments to it
      if (!(loadingStatementType & statementType::blockStatements)) {
        pushComments();
        return smnt;
      }

      // We need to create a block statement to hold these statements
      blockStatement *blockSmnt = new blockStatement(smnt->up,
                                                     smnt->source);
      statementArray &childStatements = blockSmnt->children;

      // Set the new block statement to load up comments
      childStatements.swap(comments);
      childStatements.push(smnt);

      const int childStatementCount = (int) childStatements.length();
      for (int i = 0; i < childStatementCount; ++i) {
        // Update new parent statement
        childStatements[i]->up = blockSmnt;
      }

      return blockSmnt;
    }

    statement_t* parser_t::loadBlockStatement(attributeTokenMap &smntAttributes) {
      blockStatement *smnt = new blockStatement(smntContext.up,
                                                tokenContext[0]);
      addAttributesTo(smntAttributes, smnt);

      tokenContext.pushPairRange();
      smntContext.pushUp(*smnt);
      loadAllStatements();
      smntContext.popUp();
      tokenContext.popAndSkip();
      if (!success) {
        delete smnt;
        return NULL;
      }

      return smnt;
    }

    statement_t* parser_t::loadEmptyStatement(attributeTokenMap &smntAttributes) {
      statement_t *smnt = new emptyStatement(smntContext.up,
                                             tokenContext[0]);
      addAttributesTo(smntAttributes, smnt);

      // Skip [;] token
      ++tokenContext;
      return smnt;
    }

    statement_t* parser_t::loadExpressionStatement(attributeTokenMap &smntAttributes) {
      int end = tokenContext.getNextOperator(operatorType::semicolon);
      if (end < 0) {
        if (checkSemicolon) {
          tokenContext.printErrorAtEnd("[2] Expected a [;]");
          success = false;
          return NULL;
        }
        end = tokenContext.size();
      }

      tokenContext.push(0, end);
      exprNode *expr = parseTokenContextExpression();
      tokenContext.pop();
      if (!success) {
        return NULL;
      }
      tokenContext += (end + 1);

      expressionStatement *smnt = new expressionStatement(smntContext.up,
                                                          *expr);
      addAttributesTo(smntAttributes, smnt);
      return smnt;
    }

    statement_t* parser_t::loadDeclarationStatement(attributeTokenMap &smntAttributes) {
      if (isLoadingFunction()) {
        return loadFunctionStatement(smntAttributes);
      }
      if (!success) {
        return NULL;
      }

      vartype_t baseType;
      loadBaseType(baseType);
      if (!success) {
        return NULL;
      }

      declarationStatement &smnt = *(
        new declarationStatement(smntContext.up,
                                 baseType.typeToken)
      );
      addAttributesTo(smntAttributes, &smnt);

      while(success) {
        success &= smnt.addDeclaration(
          loadVariableDeclaration(smnt.attributes, baseType)
        );
        if (!success) {
          break;
        }

        const opType_t opType = token_t::safeOperatorType(tokenContext[0]);
        if (!(opType & operatorType::comma)) {
          if (opType & operatorType::semicolon) {
            ++tokenContext;
          } else if (checkSemicolon) {
            tokenContext.printError("[3] Expected a [;]");
            success = false;
          }
          break;
        }
        ++tokenContext;
      }

      if (!success) {
        smnt.freeDeclarations();
        delete &smnt;
        return NULL;
      }

      return &smnt;
    }

    statement_t* parser_t::loadNamespaceStatement(attributeTokenMap &smntAttributes) {
      if (tokenContext.size() == 1) {
        tokenContext.printError("Expected a namespace name");
        return NULL;
      }

      // Skip [namespace] token
      ++tokenContext;
      tokenVector names;

      while (true) {
        // Get the namespace name
        if (!(tokenContext[0]->type() & tokenType::identifier)) {
          tokenContext.printError("Expected a namespace name");
          success = false;
          return NULL;
        }
        names.push_back(tokenContext[0]);

        // Check we still have a token for {
        if (tokenContext.size() == 1) {
          tokenContext.printError("Missing namespace body {}");
          success = false;
          return NULL;
        }
        ++tokenContext;

        // Find { or ::
        const opType_t opType = token_t::safeOperatorType(tokenContext[0]);
        if (!(opType & (operatorType::braceStart |
                        operatorType::scope))) {
          tokenContext.printError("Expected namespace body {}");
          success = false;
          return NULL;
        }

        if (opType & operatorType::braceStart) {
          break;
        }
        if (tokenContext.size() == 1) {
          tokenContext.printError("Missing namespace body {}");
          success = false;
          return NULL;
        }
        ++tokenContext;
      }

      namespaceStatement *smnt = NULL;
      namespaceStatement *currentSmnt = NULL;

      const int levels = (int) names.size();
      for (int i = 0; i < levels; ++i) {
        namespaceStatement *nextSmnt = new namespaceStatement(smntContext.up,
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
      tokenContext.pushPairRange();
      smntContext.pushUp(*currentSmnt);
      loadAllStatements();
      smntContext.popUp();
      tokenContext.popAndSkip();

      return smnt;
    }

    statement_t* parser_t::loadFunctionStatement(attributeTokenMap &smntAttributes) {
      function_t &func = *(new function_t());
      loadFunction(func);
      if (!success) {
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
      if (token_t::safeOperatorType(tokenContext[0]) & operatorType::semicolon) {
        ++tokenContext;
        return new functionStatement(smntContext.up,
                                     func);
      }

      // Duplicate function declared
      if (smntContext.up->hasDirectlyInScope(func.name())) {
        success = false;
        delete &func;
        return NULL;
      }

      // func() {...} <-- function declaration
      functionDeclStatement &funcSmnt = *(new functionDeclStatement(smntContext.up,
                                                                    func));
      addAttributesTo(smntAttributes, &funcSmnt);

      smntContext.pushUp(funcSmnt);
      statement_t *content = getNextStatement();
      smntContext.popUp();
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
      // when the tokenContext is at
      //     if, for, while, switch

      // Need to make sure we have another token (even if it's not '(')
      bool error = (tokenContext.size() == 1);
      if (!error) {
        ++tokenContext;
        error = !(token_t::safeOperatorType(tokenContext[0]) & operatorType::parenthesesStart);
      }

      if (error) {
        tokenContext.printError("Expected a condition statement");
        success = false;
      }
    }

    void parser_t::loadConditionStatements(statementArray &statements,
                                           const int expectedCount) {
      // Load expression/declaration
      token_t *parenBegin = tokenContext[0];
      tokenContext.pushPairRange();

      int count = 0;
      bool error = true;
      while (true) {
        const int sType = peek();
        if (!success) {
          error = true;
          break;
        }
        if (sType & statementType::none) {
          error = false;
          break;
        }

        if (sType & statementType::comment) {
          // Ignore the comment
          ++tokenContext;
          continue;
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
          tokenContext.printError(message);
          error = true;
          break;
        }

        checkSemicolon = (count < expectedCount);
        statement_t *smnt = getNextStatement();
        statements.push(smnt);
        if (!success) {
          error = true;
          break;
        }
      }
      tokenContext.popAndSkip();

      if (!error &&
          (peek() & statementType::attribute)) {
        // TODO: Support multi-location errors and point to
        //         parenEnd as a suggestion
        tokenContext.printError("Attributes should be placed as an additional statement"
                                " (e.g. [for (;;; @attr)] or [if (; @attr)])");
        error = true;
      }
      if (!success) {
        return;
      }

      const int smntCount = (int) statements.length();
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
      statementArray statements;
      loadConditionStatements(statements, 1);
      if (!statements.length()) {
        return NULL;
      }
      return statements[0];
    }

    statement_t* parser_t::loadIfStatement(attributeTokenMap &smntAttributes) {
      token_t *ifToken = tokenContext[0];
      checkIfConditionStatementExists();
      if (!success) {
        return NULL;
      }

      ifStatement &ifSmnt = *(new ifStatement(smntContext.up,
                                              ifToken));
      smntContext.pushUp(ifSmnt);
      addAttributesTo(smntAttributes, &ifSmnt);

      statement_t *condition = loadConditionStatement();
      if (!condition) {
        if (success) {
          success = false;
          tokenContext.printError("Missing condition for [if] statement");
        }
        smntContext.popUp();
        delete &ifSmnt;
        return NULL;
      }

      ifSmnt.setCondition(condition);

      statement_t *content = getNextStatement();
      if (!content) {
        if (success) {
          tokenContext.printError("Missing content for [if] statement");
          success = false;
        }
        smntContext.popUp();
        delete &ifSmnt;
        return NULL;
      }
      ifSmnt.set(*content);

      int sType;
      while ((sType = peek()) & (statementType::elif_ |
                                 statementType::else_)) {
        smntContext.pushUp(ifSmnt);
        statement_t *elSmnt = getNextStatement();
        smntContext.popUp();
        if (!elSmnt) {
          if (success) {
            break;
          }
          delete &ifSmnt;
          smntContext.popUp();
          return NULL;
        }
        if (sType & statementType::elif_) {
          ifSmnt.addElif(elSmnt->to<elifStatement>());
        } else {
          ifSmnt.addElse(elSmnt->to<elseStatement>());
          break;
        }
      }

      if (!success) {
        delete &ifSmnt;
        smntContext.popUp();
        return NULL;
      }

      smntContext.popUp();
      return &ifSmnt;
    }

    statement_t* parser_t::loadElifStatement(attributeTokenMap &smntAttributes) {
      // Skip [else] since checkIfConditionStatementExists
      //   expects 1 token before the condition
      // This is basically the same code as loadIfStatement
      //   but with an elif class
      token_t *elifToken = tokenContext[0];
      ++tokenContext;
      checkIfConditionStatementExists();
      if (!success) {
        return NULL;
      }

      elifStatement &elifSmnt = *(new elifStatement(smntContext.up,
                                                    elifToken));
      smntContext.pushUp(elifSmnt);
      addAttributesTo(smntAttributes, &elifSmnt);

      statement_t *condition = loadConditionStatement();
      if (!condition) {
        if (success) {
          success = false;
          tokenContext.printError("Missing condition for [else if] statement");
        }
        smntContext.popUp();
        return NULL;
      }

      elifSmnt.setCondition(condition);

      statement_t *content = getNextStatement();
      smntContext.popUp();
      if (!content) {
        tokenContext.printError("Missing content for [else if] statement");
        success = false;
        delete &elifSmnt;
        return NULL;
      }

      elifSmnt.set(*content);
      return &elifSmnt;
    }

    statement_t* parser_t::loadElseStatement(attributeTokenMap &smntAttributes) {
      token_t *elseToken = tokenContext[0];
      ++tokenContext;

      elseStatement &elseSmnt = *(new elseStatement(smntContext.up,
                                                    elseToken));
      smntContext.pushUp(elseSmnt);
      addAttributesTo(smntAttributes, &elseSmnt);

      statement_t *content = getNextStatement();
      smntContext.popUp();
      if (!content) {
        tokenContext.printError("Missing content for [else] statement");
        success = false;
        delete &elseSmnt;
        return NULL;
      }

      elseSmnt.set(*content);
      return &elseSmnt;
    }

    statement_t* parser_t::loadForStatement(attributeTokenMap &smntAttributes) {
      token_t *forToken = tokenContext[0];
      checkIfConditionStatementExists();
      if (!success) {
        return NULL;
      }

      forStatement &forSmnt = *(new forStatement(smntContext.up,
                                                 forToken));
      smntContext.pushUp(forSmnt);
      addAttributesTo(smntAttributes, &forSmnt);

      token_t *parenEnd = tokenContext.getClosingPairToken();

      statementArray statements;
      loadConditionStatements(statements, 3);
      if (!success) {
        smntContext.popUp();
        delete &forSmnt;
        return NULL;
      }

      int count = (int) statements.length();
      // Last statement is optional
      if (count == 2) {
        ++count;
        statements.push(new emptyStatement(smntContext.up,
                                           parenEnd,
                                           false));
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
          tokenContext.printError(message);
        }
        for (int i = 0; i < count; ++i) {
          delete statements[i];
        }
        success = false;
        smntContext.popUp();
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
      smntContext.popUp();
      if (!content) {
        if (success) {
          tokenContext.printError("Missing content for [for] statement");
          success = false;
        }
        delete &forSmnt;
        return NULL;
      }

      forSmnt.set(*content);
      return &forSmnt;
    }

    statement_t* parser_t::loadWhileStatement(attributeTokenMap &smntAttributes) {
      token_t *whileToken = tokenContext[0];
      if (getKeyword(tokenContext[0]).type() & keywordType::do_) {
        return loadDoWhileStatement(smntAttributes);
      }

      checkIfConditionStatementExists();
      if (!success) {
        return NULL;
      }

      whileStatement &whileSmnt = *(new whileStatement(smntContext.up,
                                                       whileToken));
      smntContext.pushUp(whileSmnt);
      addAttributesTo(smntAttributes, &whileSmnt);

      statement_t *condition = loadConditionStatement();
      if (!condition) {
        if (success) {
          success = false;
          tokenContext.printError("Missing condition for [while] statement");
        }
        smntContext.popUp();
        delete &whileSmnt;
        return NULL;
      }

      whileSmnt.setCondition(condition);

      statement_t *content = getNextStatement();
      smntContext.popUp();
      if (!content) {
        tokenContext.printError("Missing content for [while] statement");
        success = false;
        delete &whileSmnt;
        return NULL;
      }

      whileSmnt.set(*content);
      return &whileSmnt;
    }

    statement_t* parser_t::loadDoWhileStatement(attributeTokenMap &smntAttributes) {
      token_t *doToken = tokenContext[0];
      ++tokenContext;

      whileStatement &whileSmnt = *(new whileStatement(smntContext.up,
                                                       doToken,
                                                       true));
      smntContext.pushUp(whileSmnt);
      addAttributesTo(smntAttributes, &whileSmnt);

      statement_t *content = getNextStatement();
      if (!content) {
        if (success) {
          tokenContext.printError("Missing content for [do-while] statement");
          success = false;
        }
        smntContext.popUp();
        delete &whileSmnt;
        return NULL;
      }
      whileSmnt.set(*content);

      keyword_t &nextKeyword = getKeyword(tokenContext[0]);
      if (!(nextKeyword.type() & keywordType::while_)) {
        tokenContext.printError("Expected [while] condition after [do]");
        success = false;
        delete &whileSmnt;
        return NULL;
      }

      checkIfConditionStatementExists();
      if (!success) {
        delete &whileSmnt;
        smntContext.popUp();
        return NULL;
      }

      statement_t *condition = loadConditionStatement();
      if (!condition) {
        if (success) {
          success = false;
          tokenContext.printError("Missing condition for [do-while] statement");
        }
        delete &whileSmnt;
        smntContext.popUp();
        return NULL;
      }
      whileSmnt.setCondition(condition);

      if (!(token_t::safeOperatorType(tokenContext[0]) & operatorType::semicolon)) {
        tokenContext.printError("[5] Expected a [;]");
        success = false;
        smntContext.popUp();
        delete &whileSmnt;
        return NULL;
      }
      ++tokenContext;

      return &whileSmnt;
    }

    statement_t* parser_t::loadSwitchStatement(attributeTokenMap &smntAttributes) {
      token_t *switchToken = tokenContext[0];
      checkIfConditionStatementExists();
      if (!success) {
        return NULL;
      }

      switchStatement &switchSmnt = *(new switchStatement(smntContext.up,
                                                          switchToken));
      smntContext.pushUp(switchSmnt);
      addAttributesTo(smntAttributes, &switchSmnt);

      token_t *parenEnd = tokenContext.getClosingPairToken();
      statement_t *condition = loadConditionStatement();
      if (!condition) {
        if (success) {
          success = false;
          tokenContext.printError("Missing condition for [switch] statement");
        }
        smntContext.popUp();
        delete &switchSmnt;
        return NULL;
      }

      switchSmnt.setCondition(condition);

      statement_t *content = getNextStatement();
      smntContext.popUp();
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
      token_t *caseToken = tokenContext[0];
      ++tokenContext;

      const int pos = tokenContext.getNextOperator(operatorType::colon);
      // No : found
      if (pos < 0) {
        tokenContext.printError("Expected a [:] to close the [case] statement");
        success = false;
        return NULL;
      }
      exprNode *value = NULL;
      // The case where we see 'case:'
      if (0 < pos) {
        // Load the case expression
        value = parseTokenContextExpression(0, pos);
      }
      if (!value) {
        tokenContext.printError("Expected a constant expression for the [case] statement");
        success = false;
        return NULL;
      }

      tokenContext += (pos + 1);

      caseStatement *smnt = new caseStatement(smntContext.up,
                                              caseToken,
                                              *value);
      addAttributesTo(smntAttributes, smnt);
      return smnt;
    }

    statement_t* parser_t::loadDefaultStatement(attributeTokenMap &smntAttributes) {
      token_t *defaultToken = tokenContext[0];
      ++tokenContext;
      if (!(token_t::safeOperatorType(tokenContext[0]) & operatorType::colon)) {
        tokenContext.printError("Expected a [:]");
        success = false;
        return NULL;
      }
      ++tokenContext;

      defaultStatement *smnt = new defaultStatement(smntContext.up,
                                                    defaultToken);
      addAttributesTo(smntAttributes, smnt);
      return smnt;
    }

    statement_t* parser_t::loadContinueStatement(attributeTokenMap &smntAttributes) {
      token_t *continueToken = tokenContext[0];
      ++tokenContext;
      if (!(token_t::safeOperatorType(tokenContext[0]) & operatorType::semicolon)) {
        tokenContext.printError("[6] Expected a [;]");
        success = false;
        return NULL;
      }
      ++tokenContext;

      continueStatement *smnt = new continueStatement(smntContext.up,
                                                      continueToken);
      addAttributesTo(smntAttributes, smnt);
      return smnt;
    }

    statement_t* parser_t::loadBreakStatement(attributeTokenMap &smntAttributes) {
      token_t *breakToken = tokenContext[0];
      ++tokenContext;
      if (!(token_t::safeOperatorType(tokenContext[0]) & operatorType::semicolon)) {
        tokenContext.printError("[7] Expected a [;]");
        success = false;
        return NULL;
      }
      ++tokenContext;

      breakStatement *smnt = new breakStatement(smntContext.up,
                                                breakToken);
      addAttributesTo(smntAttributes, smnt);
      return smnt;
    }

    statement_t* parser_t::loadReturnStatement(attributeTokenMap &smntAttributes) {
      token_t *returnToken = tokenContext[0];
      ++tokenContext;

      const int pos = tokenContext.getNextOperator(operatorType::semicolon);
      // No ; found
      if (pos < 0) {
        tokenContext.printErrorAtEnd("[8] Expected a [;]");
        success = false;
        return NULL;
      }
      exprNode *value = NULL;
      // The case where we see 'return;'
      if (0 < pos) {
        // Load the return value
        value = parseTokenContextExpression(0, pos);
        if (!success) {
          return NULL;
        }
      }

      tokenContext += (pos + 1);

      returnStatement *smnt = new returnStatement(smntContext.up,
                                                  returnToken,
                                                  value);
      addAttributesTo(smntAttributes, smnt);
      return smnt;
    }

    statement_t* parser_t::loadClassAccessStatement(attributeTokenMap &smntAttributes) {
      token_t *accessToken = tokenContext[0];
      if (!(token_t::safeOperatorType(tokenContext[1]) & operatorType::colon)) {
        tokenContext.printError("Expected a [:]");
        success = false;
        return NULL;
      }
      const int kType = getKeyword(tokenContext[0]).type();
      tokenContext += 2;

      int access = classAccess::private_;
      if (kType == keywordType::public_) {
        access = classAccess::public_;
      } else if (kType == keywordType::protected_) {
        access = classAccess::protected_;
      }

      classAccessStatement *smnt = new classAccessStatement(smntContext.up,
                                                            accessToken,
                                                            access);
      addAttributesTo(smntAttributes, smnt);
      return smnt;
    }

    statement_t* parser_t::loadDirectiveStatement(attributeTokenMap &smntAttributes) {
      directiveStatement *smnt = new directiveStatement(smntContext.up,
                                                        *((directiveToken*) tokenContext[0]));
      addAttributesTo(smntAttributes, smnt);

      ++tokenContext;

      return smnt;
    }

    statement_t* parser_t::loadPragmaStatement(attributeTokenMap &smntAttributes) {
      pragmaStatement *smnt = new pragmaStatement(smntContext.up,
                                                  *((pragmaToken*) tokenContext[0]));
      addAttributesTo(smntAttributes, smnt);

      ++tokenContext;

      return smnt;
    }

    statement_t* parser_t::loadGotoStatement(attributeTokenMap &smntAttributes) {
      ++tokenContext;
      if (!(token_t::safeType(tokenContext[0]) & tokenType::identifier)) {
        tokenContext.printError("Expected [goto label] identifier");
        success = false;
        return NULL;
      }
      if (!(token_t::safeOperatorType(tokenContext[1]) & operatorType::semicolon)) {
        tokenContext.printError("[9] Expected a [;]");
        success = false;
        return NULL;
      }

      identifierToken &labelToken = (tokenContext[0]
                                     ->clone()
                                     ->to<identifierToken>());
      tokenContext += 2;

      gotoStatement *smnt = new gotoStatement(smntContext.up,
                                              labelToken);
      addAttributesTo(smntAttributes, smnt);
      return smnt;
    }

    statement_t* parser_t::loadGotoLabelStatement(attributeTokenMap &smntAttributes) {
      if (!(token_t::safeOperatorType(tokenContext[1]) & operatorType::colon)) {
        tokenContext.printError("Expected a [:]");
        success = false;
        return NULL;
      }

      identifierToken &labelToken = (tokenContext[0]
                                     ->clone()
                                     ->to<identifierToken>());
      tokenContext += 2;

      gotoLabelStatement *smnt = new gotoLabelStatement(smntContext.up,
                                                        labelToken);
      addAttributesTo(smntAttributes, smnt);
      return smnt;
    }
    //==================================
  }
}
