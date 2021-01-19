#include <sstream>
#include <stdlib.h>

#include <occa/utils/hash.hpp>
#include <occa/internal/io.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/utils/lex.hpp>
#include <occa/internal/utils/string.hpp>
#include <occa/types/primitive.hpp>

#include <occa/internal/lang/preprocessor.hpp>
#include <occa/internal/lang/specialMacros.hpp>
#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/tokenizer.hpp>

namespace occa {
  namespace lang {
    namespace ppStatus {
      const int reading    = (1 << 0);
      const int ignoring   = (1 << 1);
      const int foundIf    = (1 << 2);
      const int foundElse  = (1 << 3);
      const int finishedIf = (1 << 4);
    }

    // TODO: Add actual compiler macros as well
    preprocessor_t::preprocessor_t(const occa::json &settings_) {
      init();
      initDirectives();
      initStandardHeaders();

      setSettings(settings_);

      includePaths = env::OCCA_INCLUDE_PATH;

      strictHeaders = settings.get("okl/strict_headers", true);

      json oklIncludePaths = settings["okl/include_paths"];
      if (oklIncludePaths.isArray()) {
        jsonArray pathArray = oklIncludePaths.array();
        const int pathCount = (int) pathArray.size();
        for (int i = 0; i < pathCount; ++i) {
          json path = pathArray[i];
          if (path.isString()) {
            includePaths.push_back(path);
          }
        }
      }

      const int includePathCount = (int) includePaths.size();
      for (int i = 0; i < includePathCount; ++i) {
        io::endWithSlash(includePaths[i]);
      }
    }

    preprocessor_t::preprocessor_t(const preprocessor_t &pp) :
      withCache(pp) {
      *this = pp;
    }

    preprocessor_t::~preprocessor_t() {
      clear_();
    }

    void preprocessor_t::init() {
      pushStatus(ppStatus::reading);
      // Always start off as if we passed a newline
      incrementNewline();
      expandingMacros = true;

      const int specialMacroCount = 8;
      macro_t *specialMacros[specialMacroCount] = {
        new definedMacro(*this),    // defined()
        new hasIncludeMacro(*this), // __has_include()
        new fileMacro(*this),       // __FILE__
        new lineMacro(*this),       // __LINE__
        new dateMacro(*this),       // __DATE__
        new timeMacro(*this),       // __TIME__
        new counterMacro(*this),    // __COUNTER__
        new oklMacro(*this)         // OKL
      };
      for (int i = 0; i < specialMacroCount; ++i) {
        compilerMacros[specialMacros[i]->name()] = specialMacros[i];
      }

      // OCCA-specific macros
      addCompilerDefine("OCCA_MAJOR_VERSION", toString(OCCA_MAJOR_VERSION));
      addCompilerDefine("OCCA_MINOR_VERSION", toString(OCCA_MINOR_VERSION));
      addCompilerDefine("OCCA_PATCH_VERSION", toString(OCCA_PATCH_VERSION));
      addCompilerDefine("OCCA_VERSION", toString(OCCA_VERSION));
      addCompilerDefine("OKL_VERSION" , toString(OKL_VERSION));
      addCompilerDefine("__OKL__" , "1");
      addCompilerDefine("__OCCA__" , "1");

      const std::string mode = settings.get<std::string>("mode", "");
      addCompilerDefine("OKL_MODE", "\"" + uppercase(mode) + "\"");

      // Add kernel hash as a define
      std::string hashValue = settings.get<std::string>("hash", "");
      if (hashValue.size()) {
        hash_t hash = hash_t::fromString(hashValue);
        hashValue = hash.getString();
      } else {
        hashValue = "unknown";
      }
      addCompilerDefine("OKL_KERNEL_HASH", "\"" + hashValue + "\"");

      // Alternative representations
      addCompilerDefine("and"   , "&&");
      addCompilerDefine("and_eq", "&=");
      addCompilerDefine("bitand", "&");
      addCompilerDefine("bitor" , "|");
      addCompilerDefine("compl" , "~");
      addCompilerDefine("not"   , "!");
      addCompilerDefine("not_eq", "!=");
      addCompilerDefine("or"    , "||");
      addCompilerDefine("or_eq" , "|=");
      addCompilerDefine("xor"   , "^");
      addCompilerDefine("xor_eq", "^=");

      warnings = 0;
      errors   = 0;

      tokenizer = NULL;
    }

    void preprocessor_t::clear() {
      clear_();
      init();
    }

    void preprocessor_t::clear_() {
      errors   = 0;
      warnings = 0;

      tokenizer = NULL;

      while (inputCache.size()) {
        delete inputCache.front();
        inputCache.pop_front();
      }
      while (outputCache.size()) {
        delete outputCache.front();
        outputCache.pop_front();
      }

      statusStack.clear();

      macroMap::iterator it = compilerMacros.begin();
      while (it != compilerMacros.end()) {
        delete it->second;
        ++it;
      }
      compilerMacros.clear();

      it = sourceMacros.begin();
      while (it != sourceMacros.end()) {
        delete it->second;
        ++it;
      }
      sourceMacros.clear();

      dependencies.clear();
    }

    preprocessor_t& preprocessor_t::operator = (const preprocessor_t &other) {
      clear();

      statusStack     = other.statusStack;
      status          = other.status;
      passedNewline   = other.passedNewline;
      expandingMacros = other.expandingMacros;

      directives     = other.directives;
      compilerMacros = other.compilerMacros;
      sourceMacros   = other.sourceMacros;

      dependencies = other.dependencies;
      warnings     = other.warnings;
      errors       = other.errors;

      includePaths = other.includePaths;

      // Copy cache
      tokenList *caches[2] = { &inputCache, &outputCache };
      const tokenList *otherCaches[2] = { &other.inputCache, &other.outputCache };
      for (int i = 0; i < 2; ++i) {
        tokenList &cache            = *(caches[i]);
        const tokenList &otherCache = *(otherCaches[i]);

        tokenList::const_iterator it = otherCache.begin();
        while (it != otherCache.end()) {
          cache.push_back((*it)->clone());
          ++it;
        }
      }

      // Copy macros
      compilerMacros = other.compilerMacros;
      sourceMacros   = other.sourceMacros;
      macroMap::iterator it = compilerMacros.begin();
      while (it != compilerMacros.end()) {
        it->second = &(it->second->clone(*this));
        ++it;
      }
      it = sourceMacros.begin();
      while (it != compilerMacros.end()) {
        it->second = &(it->second->clone(*this));
        ++it;
      }
      return *this;
    }

    void preprocessor_t::setSettings(occa::json settings_) {
      settings = settings_;
    }

    void preprocessor_t::initDirectives() {
      directives["if"]      = &preprocessor_t::processIf;
      directives["ifdef"]   = &preprocessor_t::processIfdef;
      directives["ifndef"]  = &preprocessor_t::processIfndef;
      directives["elif"]    = &preprocessor_t::processElif;
      directives["else"]    = &preprocessor_t::processElse;
      directives["endif"]   = &preprocessor_t::processEndif;

      directives["define"]  = &preprocessor_t::processDefine;
      directives["undef"]   = &preprocessor_t::processUndef;

      directives["error"]   = &preprocessor_t::processError;
      directives["warning"] = &preprocessor_t::processWarning;

      directives["include"] = &preprocessor_t::processInclude;
      directives["pragma"]  = &preprocessor_t::processPragma;
      directives["line"]    = &preprocessor_t::processLine;
    }

    void preprocessor_t::initStandardHeaders() {
      standardHeaders.insert("algorithm");
      standardHeaders.insert("any");
      standardHeaders.insert("array");
      standardHeaders.insert("assert.h");
      standardHeaders.insert("atomic");
      standardHeaders.insert("bitset");
      standardHeaders.insert("cassert");
      standardHeaders.insert("ccomplex");
      standardHeaders.insert("cctype");
      standardHeaders.insert("cerrno");
      standardHeaders.insert("cfenv");
      standardHeaders.insert("cfloat");
      standardHeaders.insert("charconv");
      standardHeaders.insert("chrono");
      standardHeaders.insert("cinttypes");
      standardHeaders.insert("ciso646");
      standardHeaders.insert("climits");
      standardHeaders.insert("clocale");
      standardHeaders.insert("cmath");
      standardHeaders.insert("codecvt");
      standardHeaders.insert("compare");
      standardHeaders.insert("complex.h");
      standardHeaders.insert("complex");
      standardHeaders.insert("condition_variable");
      standardHeaders.insert("csetjmp");
      standardHeaders.insert("csignal");
      standardHeaders.insert("cstdalign");
      standardHeaders.insert("cstdarg");
      standardHeaders.insert("cstdbool");
      standardHeaders.insert("cstddef");
      standardHeaders.insert("cstdint");
      standardHeaders.insert("cstdio");
      standardHeaders.insert("cstdlib");
      standardHeaders.insert("cstring");
      standardHeaders.insert("ctgmath");
      standardHeaders.insert("ctime");
      standardHeaders.insert("ctype.h");
      standardHeaders.insert("cuchar");
      standardHeaders.insert("cwchar");
      standardHeaders.insert("cwctype");
      standardHeaders.insert("deque");
      standardHeaders.insert("errno.h");
      standardHeaders.insert("exception");
      standardHeaders.insert("execution");
      standardHeaders.insert("fenv.h");
      standardHeaders.insert("filesystem");
      standardHeaders.insert("float.h");
      standardHeaders.insert("forward_list");
      standardHeaders.insert("fstream");
      standardHeaders.insert("functional");
      standardHeaders.insert("future");
      standardHeaders.insert("initializer_list");
      standardHeaders.insert("inttypes.h");
      standardHeaders.insert("iomanip");
      standardHeaders.insert("ios");
      standardHeaders.insert("iosfwd");
      standardHeaders.insert("iostream");
      standardHeaders.insert("iso646.h");
      standardHeaders.insert("istream");
      standardHeaders.insert("iterator");
      standardHeaders.insert("limits.h");
      standardHeaders.insert("limits");
      standardHeaders.insert("list");
      standardHeaders.insert("locale.h");
      standardHeaders.insert("locale");
      standardHeaders.insert("map");
      standardHeaders.insert("math.h");
      standardHeaders.insert("memory");
      standardHeaders.insert("memory_resource");
      standardHeaders.insert("mutex");
      standardHeaders.insert("new");
      standardHeaders.insert("numeric");
      standardHeaders.insert("optional");
      standardHeaders.insert("ostream");
      standardHeaders.insert("queue");
      standardHeaders.insert("random");
      standardHeaders.insert("ratio");
      standardHeaders.insert("regex");
      standardHeaders.insert("scoped_allocator");
      standardHeaders.insert("set");
      standardHeaders.insert("setjmp.h");
      standardHeaders.insert("shared_mutex");
      standardHeaders.insert("signal.h");
      standardHeaders.insert("sstream");
      standardHeaders.insert("stack");
      standardHeaders.insert("stdalign.h");
      standardHeaders.insert("stdarg.h");
      standardHeaders.insert("stdatomic.h");
      standardHeaders.insert("stdbool.h");
      standardHeaders.insert("stddef.h");
      standardHeaders.insert("stdexcept");
      standardHeaders.insert("stdint.h");
      standardHeaders.insert("stdio.h");
      standardHeaders.insert("stdlib.h");
      standardHeaders.insert("stdnoreturn.h");
      standardHeaders.insert("streambuf");
      standardHeaders.insert("string.h");
      standardHeaders.insert("string");
      standardHeaders.insert("string_view");
      standardHeaders.insert("strstream");
      standardHeaders.insert("syncstream");
      standardHeaders.insert("system_error");
      standardHeaders.insert("tgmath.h");
      standardHeaders.insert("thread");
      standardHeaders.insert("threads.h");
      standardHeaders.insert("time.h");
      standardHeaders.insert("tuple");
      standardHeaders.insert("type_traits");
      standardHeaders.insert("typeindex");
      standardHeaders.insert("typeinfo");
      standardHeaders.insert("uchar.h");
      standardHeaders.insert("unordered_map");
      standardHeaders.insert("unordered_set");
      standardHeaders.insert("utility");
      standardHeaders.insert("valarray");
      standardHeaders.insert("variant");
      standardHeaders.insert("vector");
      standardHeaders.insert("wchar.h");
      standardHeaders.insert("wctype.h");
    }

    void preprocessor_t::warningOn(token_t *token,
                                   const std::string &message) {
      ++warnings;
      token->printWarning(message);
    }

    void preprocessor_t::errorOn(token_t *token,
                                 const std::string &message) {
      ++errors;
      token->printError(message);
    }

    tokenMap& preprocessor_t::clone_() const {
      return *(new preprocessor_t(*this));
    }

    void* preprocessor_t::passMessageToInput(const occa::json &props) {
      const std::string inputName = props.get<std::string>("input_name");
      if (inputName == "preprocessor_t") {
        return (void*) this;
      }
      if (input) {
        return input->passMessageToInput(props);
      }
      return NULL;
    }

    void preprocessor_t::pushStatus(const int status_) {
      statusStack.push_back(status);
      status = status_;
    }

    int preprocessor_t::popStatus() {
      if (statusStack.size() == 0) {
        return 0;
      }
      status = statusStack.back();
      statusStack.pop_back();
      return status;
    }

    void preprocessor_t::swapReadingStatus() {
      if (status & ppStatus::reading) {
        status &= ~ppStatus::reading;
        status |= ppStatus::ignoring;
      } else {
        status &= ~ppStatus::ignoring;
        status |= ppStatus::reading;
      }
    }

    void preprocessor_t::incrementNewline() {
      // We need to keep passedNewline 'truthy'
      //   until after the next token
      passedNewline = 2;
    }

    void preprocessor_t::decrementNewline() {
      passedNewline -= !!passedNewline;
    }

    macro_t* preprocessor_t::getMacro(const std::string &name) {
      macroMap::iterator it = sourceMacros.find(name);
      if (it != sourceMacros.end()) {
        return it->second;
      }
      it = compilerMacros.find(name);
      if (it != compilerMacros.end()) {
        return it->second;
      }
      return NULL;
    }

    token_t* preprocessor_t::getSourceToken() {
      token_t *token = NULL;
      if (!inputIsEmpty()) {
        getNextInput(token);
      }
      return token;
    }

    void preprocessor_t::fetchNext() {
      processToken(getSourceToken());
    }

    //---[ Public ]---------------------
    void preprocessor_t::addCompilerDefine(const std::string &name,
                                           const std::string &value) {
      removeCompilerDefine(name);
      compilerMacros[name] = macro_t::defineBuiltin(*this,
                                                    name,
                                                    value);
    }

    void preprocessor_t::removeCompilerDefine(const std::string &name) {
      macroMap::iterator mIt = compilerMacros.find(name);
      if (mIt != compilerMacros.end()) {
        delete mIt->second;
        compilerMacros.erase(mIt);
      }
    }

    void preprocessor_t::addSourceDefine(const std::string &name,
                                         const std::string &value) {
      sourceMacros[name] = macro_t::defineBuiltin(*this,
                                                  name,
                                                  value);
    }

    void preprocessor_t::removeSourceDefine(const std::string &name) {
      macroMap::iterator mIt = sourceMacros.find(name);
      if (mIt != sourceMacros.end()) {
        delete mIt->second;
        sourceMacros.erase(mIt);
      }
    }

    strVector preprocessor_t::getDependencyFilenames() const {
      const int dependencyCount = dependencies.size();
      strVector deps;
      deps.reserve(dependencyCount);

      strToBoolMap::const_iterator it = dependencies.begin();
      while (it != dependencies.end()) {
        deps.push_back(it->first);
        ++it;
      }

      return deps;
    }
    //==================================

    void preprocessor_t::loadTokenizer() {
      if (!tokenizer) {
        tokenizer = (tokenizer_t*) getInput("tokenizer_t");
      }
    }

    void preprocessor_t::injectSourceCode(token_t &sourceToken,
                                          const std::string &source,
                                          const bool addNewlineAfterSource) {
      // Make sure we have a tokenizer
      loadTokenizer();
      if (!tokenizer) {
        warningOn(&sourceToken,
                  "Unable to apply @directive due to the lack of a tokenizer");
        return;
      }

      tokenVector newTokens;
      tokenizer->tokenize(newTokens,
                          sourceToken.origin,
                          source);

      if (addNewlineAfterSource) {
        incrementNewline();
        pushInput(new newlineToken(sourceToken.origin));
      }
      const int newTokenCount = (int) newTokens.size();
      for (int i = (newTokenCount - 1); i >= 0; --i) {
        pushInput(newTokens[i]);
      }
    }

    bool preprocessor_t::expandDefinedToken(token_t *token, tokenVector &tokens) {
      if (!(token_t::safeType(token) & tokenType::identifier)) {
        return false;
      }

      identifierToken &idToken = token->to<identifierToken>();
      macro_t *macro = getMacro(idToken.value);
      if (!macro) {
        return false;
      }

      macro->expand(tokens, idToken);
      return true;
    }

    void preprocessor_t::expandDefinedTokens(tokenVector &inputTokens,
                                             tokenVector &outputTokens) {
      const int inputTokenCount = (int) inputTokens.size();
      for (int i = 0; i < inputTokenCount; ++i) {
        token_t *token = inputTokens[i];
        tokenVector expandedTokens;

        if (expandDefinedToken(token, expandedTokens)) {
          const int expandedTokenCount = (int) expandedTokens.size();
          for (int j = 0; j < expandedTokenCount; ++j) {
            outputTokens.push_back(expandedTokens[j]);
          }
        } else {
          outputTokens.push_back(token->clone());
        }
      }
    }

    void preprocessor_t::expandMacro(identifierToken &source,
                                     macro_t &macro) {
      tokenVector tokens;
      macro.expand(tokens, source);

      const int tokenCount = (int) tokens.size();
      if (!tokenCount) {
        return;
      }

      macroVector &tokenMacros = expandedMacroEnd[tokens[tokenCount - 1]];
      // Move tokens that would have ended here to the end of its expansion
      macroEndMap::iterator it = expandedMacroEnd.find(&source);
      if (it != expandedMacroEnd.end()) {
        tokenMacros = it->second;
        expandedMacroEnd.erase(it);
      }
      // Set expanded macro info
      expandedMacros[&macro] = true;
      tokenMacros.push_back(&macro);

      // Insert tokens backwards into input cache
      for (int i = (tokenCount - 1); i >= 0; --i) {
        pushInput(tokens[i]);
      }
    }

    void preprocessor_t::clearExpandedMacros(token_t *token) {
      macroEndMap::iterator it = expandedMacroEnd.find(token);
      if (it == expandedMacroEnd.end()) {
        return;
      }
      // Remove expanded macros at the end token
      macroVector &macros = it->second;
      const int count = (int) macros.size();
      for (int i = 0; i < count; ++i) {
        expandedMacros.erase(macros[i]);
      }
      expandedMacroEnd.erase(it);
    }

    void preprocessor_t::skipToNewline() {
      tokenVector lineTokens;
      getLineTokens(lineTokens);

      // Push the newline token
      const int tokens = (int) lineTokens.size();
      if (tokens) {
        pushOutput(lineTokens[tokens-1]);
        lineTokens.pop_back();
      }

      freeTokenVector(lineTokens);
    }

    // lineTokens might be partially initialized
    //   so we don't want to clear it
    void preprocessor_t::getLineTokens(tokenVector &lineTokens) {
      while (!inputIsEmpty()) {
        token_t *token = getSourceToken();

        if (token->type() & tokenType::newline) {
          incrementNewline();
          lineTokens.push_back(token);
          break;
        }

        lineTokens.push_back(token);
      }
    }

    // lineTokens might be partially initialized
    //   so we don't want to clear it
    void preprocessor_t::getExpandedLineTokens(tokenVector &lineTokens) {
      // Make sure we don't ignore these tokens
      int oldStatus = status;
      status = ppStatus::reading;

      while (!inputIsEmpty()) {
        token_t *token = NULL;
        (*this) >> token;

        if (token->type() & tokenType::newline) {
          incrementNewline();
          lineTokens.push_back(token);
          status = oldStatus;
          break;
        }

        lineTokens.push_back(token);
      }
      status = oldStatus;
    }

    void preprocessor_t::warnOnNonEmptyLine(const std::string &message) {
      tokenVector lineTokens;
      getLineTokens(lineTokens);
      if (lineTokens.size()) {
        // Don't account for the newline token
        if (!(lineTokens[0]->type() & (tokenType::newline |
                                       tokenType::comment))) {
          warningOn(lineTokens[0],
                    message);
        }
        freeTokenVector(lineTokens);
      }
    }

    void preprocessor_t::removeNewline(tokenVector &lineTokens) {
      const int tokens = (int) lineTokens.size();
      if (!tokens) {
        return;
      }
      token_t *lastToken = lineTokens[tokens - 1];
      if (lastToken->type() & tokenType::newline) {
        delete lastToken;
        lineTokens.pop_back();
      }
    }

    void preprocessor_t::processToken(token_t *token) {
      decrementNewline();

      const int tokenType = token->type();

      if (tokenType & tokenType::newline) {
        incrementNewline();
        pushOutput(token);
        clearExpandedMacros(token);
        return;
      }

      // Only process operators when ignoring
      //   for potential #
      if (status & ppStatus::ignoring) {
        if (!canProcessWhileIgnoring(token)) {
          clearExpandedMacros(token);
          delete token;
          return;
        }
      }

      if (tokenType & tokenType::identifier) {
        processIdentifier(token->to<identifierToken>());
      }
      else if (tokenType & tokenType::op) {
        processOperator(token->to<operatorToken>());
      }
      else {
        pushOutput(token);
      }

      clearExpandedMacros(token);
    }

    bool preprocessor_t::canProcessWhileIgnoring(token_t *token) {
      opType_t opType = token->getOpType();
      // Can process # directive to stop an #if/#else/#elif
      if (opType & operatorType::hash) {
        return true;
      }
      // Can process @directive directives
      if (opType & operatorType::attribute) {
        token_t *directiveToken;
        const bool isDirective = (
          processingDirectiveAttribute(token->to<operatorToken>(),
                                       directiveToken)
        );
        // Push the directive token as well
        if (directiveToken) {
          pushInput(directiveToken);
        }

        return isDirective;
      }

      return false;
    }

    bool preprocessor_t::processingDirectiveAttribute(operatorToken &opToken,
                                                      token_t *&directiveToken) {
      // Check for @[directive]
      directiveToken = getSourceToken();
      return (
        (token_t::safeType(directiveToken) & tokenType::identifier) &&
        (directiveToken->to<identifierToken>().value == "directive")
      );
    }

    void preprocessor_t::processIdentifier(identifierToken &token) {
      // Ignore tokens inside disabled #if/#elif/#else regions
      if (status & ppStatus::ignoring) {
        delete &token;
        return;
      }

      macro_t *macro = (expandingMacros
                        ? getMacro(token.value)
                        : NULL);

      // Don't allow for recursive expansion
      if (!macro
          || (expandedMacros.find(macro) != expandedMacros.end())) {
        pushOutput(&token);
        return;
      }
      // Check for the type of macro
      if (!macro->isFunctionLike) {
        expandMacro(token, *macro);
        delete &token;
        return;
      }
      // Function-like macro is not called
      if (inputIsEmpty()) {
        pushOutput(&token);
        return;
      }
      // Make sure that the macro starts with a '('
      token_t *nextToken = NULL;
      (*this) >> nextToken;
      if (nextToken->getOpType() & operatorType::parenthesesStart) {
        expandMacro(token, *macro);
        delete &token;
        delete nextToken;
        return;
      }
      // Prioritize possible variable if no () is found:
      //   #define FOO()
      //   int FOO;
      pushOutput(&token);
      if (nextToken) {
        pushInput(nextToken);
      }
    }

    void preprocessor_t::processOperator(operatorToken &opToken) {
      const opType_t &op = opToken.opType();
      // Make sure the line starts with #
      if ((op == operatorType::hash) && passedNewline) {
        return processHashOperator(opToken);
      }
      // Test for @directive("...")
      if (op == operatorType::attribute) {
        return processAttributeOperator(opToken);
      }
      pushOutput(&opToken);
    }

    void preprocessor_t::processHashOperator(operatorToken &opToken) {
      delete &opToken;
      if (inputIsEmpty()) {
        return;
      }

      // NULL or an empty # is ok
      token_t *directive = getSourceToken();
      if (directive->type() & tokenType::newline) {
        incrementNewline();
        pushOutput(directive);
        return;
      }

      // Check for valid directive
      if (directive->type() != tokenType::identifier) {
        errorOn(directive,
                "Unknown preprocessor directive");
        skipToNewline();
        return;
      }

      identifierToken &directiveToken = directive->to<identifierToken>();
      const std::string &directiveStr = directiveToken.value;
      directiveMap::iterator it = directives.find(directiveStr);
      if (it == directives.end()) {
        errorOn(directive,
                "Unknown preprocessor directive");
        delete directive;
        skipToNewline();
        return;
      }

      processDirective_t processFunc = it->second;

      if ((status & ppStatus::ignoring)                   &&
          (processFunc != &preprocessor_t::processIf)     &&
          (processFunc != &preprocessor_t::processIfdef)  &&
          (processFunc != &preprocessor_t::processIfndef) &&
          (processFunc != &preprocessor_t::processElif)   &&
          (processFunc != &preprocessor_t::processElse)   &&
          (processFunc != &preprocessor_t::processEndif)) {

        delete directive;
        skipToNewline();
        return;
      }

      (this->*(processFunc))(directiveToken);

      delete directive;
    }

    void preprocessor_t::processAttributeOperator(operatorToken &opToken) {
      token_t *directiveToken;
      if (!processingDirectiveAttribute(opToken, directiveToken)) {
        // Push the [@] token directly
        pushOutput(&opToken);
        // The next token still needs to be processed
        if (directiveToken) {
          pushInput(directiveToken);
        }
        return;
      }

      // Freeze our outputs and expand the rest of our symbols
      std::list<token_t*> prevOutputCache = outputCache;
      pushStatus(ppStatus::reading);
      outputCache.clear();

      // Make sure we have a tokenizer
      loadTokenizer();
      if (!tokenizer) {
        warningOn(directiveToken,
                  "Unable to apply @directive due to the lack of a tokenizer");
        popStatus();
        return freeAttributeOperatorTokens(opToken,
                                           *directiveToken,
                                           prevOutputCache);
      }

      // Try and get the 3 @directive[(]["..."][)] tokens
      const int currentTokenizerErrors = tokenizer->errors;
      const int currentErrors = errors;
      while (!inputIsEmpty() && outputCache.size() < 3) {
        fetchNext();
        if ((errors > currentErrors) ||
            (tokenizer->errors > currentTokenizerErrors)) {
          popStatus();
          return freeAttributeOperatorTokens(opToken,
                                             *directiveToken,
                                             prevOutputCache);
        }
      }

      if (outputCache.size() < 3) {
        errorOn(directiveToken,
                "[1] @directive expects a string argument");
        popStatus();
        return freeAttributeOperatorTokens(opToken,
                                           *directiveToken,
                                           prevOutputCache);
      }

      // Check for the proper 3 @directive[(]["..."][)] tokens
      token_t *parenStartToken = outputCache.front();
      outputCache.pop_front();
      token_t *contentToken = outputCache.front();
      outputCache.pop_front();
      token_t *parenEndToken = outputCache.front();
      outputCache.pop_front();

      if (
        !(token_t::safeOperatorType(parenStartToken) & operatorType::parenthesesStart)
        || !(token_t::safeType(contentToken) & tokenType::string)
        || !(token_t::safeOperatorType(parenEndToken) & operatorType::parenthesesEnd)
      ) {
        errorOn(directiveToken,
                "[2] @directive expects a string argument");
        delete parenStartToken;
        delete contentToken;
        delete parenEndToken;
        skipToNewline();
        popStatus();
        return freeAttributeOperatorTokens(opToken,
                                           *directiveToken,
                                           prevOutputCache);
      }

      stringToken &sourceToken = contentToken->to<stringToken>();
      const std::string source = strip(sourceToken.value);
      const int sourceLength = (int) source.size();

      // Make sure @directive content starts with a #
      bool missingStartHash = (!sourceLength || source[0] != '#');
      // Make sure there are no newlines inserted
      bool hasNewlines = (
        !missingStartHash
        && source.find("\\n") != std::string::npos
      );

      // Print error message if needed
      if (missingStartHash || hasNewlines) {
        if (missingStartHash) {
          errorOn(contentToken,
                  "@directive expects a string argument which starts with #");
        } else {
          errorOn(contentToken,
                  "@directive expects a string argument cannot contain newlines");
        }
        delete parenStartToken;
        delete contentToken;
        delete parenEndToken;
        popStatus();
        return freeAttributeOperatorTokens(opToken,
                                           *directiveToken,
                                           prevOutputCache);
      }

      injectSourceCode(sourceToken, source);

      delete &opToken;
      delete directiveToken;

      // Bring back the original output cache
      outputCache = prevOutputCache;
      popStatus();
    }

    void preprocessor_t::freeAttributeOperatorTokens(token_t &opToken,
                                                     token_t &directiveToken,
                                                     std::list<token_t*> &prevOutputCache) {
      delete &opToken;
      delete &directiveToken;

      // Delete output tokens
      while (!outputCache.empty()) {
        delete outputCache.front();
        outputCache.pop_front();
      }
      while (!prevOutputCache.empty()) {
        delete prevOutputCache.front();
        prevOutputCache.pop_front();
      }
    }

    bool preprocessor_t::lineIsTrue(identifierToken &directive,
                                    bool &isTrue) {
      tokenVector lineTokens;
      getExpandedLineTokens(lineTokens);
      removeNewline(lineTokens);

      const int tokenCount = (int) lineTokens.size();
      for (int i = 0; i < tokenCount; ++i) {
        token_t *token = lineTokens[i];
        if (!(token->type() & tokenType::identifier)) {
          continue;
        }
        lineTokens[i] = new primitiveToken(token->origin, 0, "0");
        delete token;
      }

      exprNode *expr = expressionParser::parse(lineTokens);

      // Errors when expr is NULL are handled
      //   while forming the expression
      bool exprError = !expr;
      if (expr) {
        if (expr->type() & exprNodeType::empty) {
          errorOn(&directive,
                  "Expected a value or expression");
          exprError = true;
        }
        else if (!expr->canEvaluate()) {
          errorOn(&directive,
                  "Unable to evaluate expression");
          exprError = true;
        }
      }

      // Default to #if false with error
      if (exprError) {
        pushStatus(ppStatus::ignoring |
                   ppStatus::foundIf);
        return false;
      }

      isTrue = expr->evaluate();
      delete expr;

      return true;
    }

    bool preprocessor_t::getIfdef(identifierToken &directive,
                                  bool &isTrue) {
      token_t *token = getSourceToken();
      const int tokenType = token_t::safeType(token);

      if (!(tokenType & tokenType::identifier)) {
        // Print from the directive if we don't
        //   have a token in the same line
        token_t *errorToken = &directive;
        if (tokenType & tokenType::newline) {
          incrementNewline();
          pushOutput(token);
        } else if (tokenType & ~tokenType::none) {
          errorToken = token;
        }
        errorOn(errorToken,
                "Expected an identifier");
        delete token;

        // Default to false
        pushStatus(ppStatus::ignoring |
                   ppStatus::foundIf);
        return false;
      }

      const std::string &macroName = token->to<identifierToken>().value;
      isTrue = getMacro(macroName);
      delete token;

      return true;
    }

    void preprocessor_t::processIf(identifierToken &directive) {
      // Nested case
      if (status & ppStatus::ignoring) {
        skipToNewline();
        pushStatus(ppStatus::ignoring |
                   ppStatus::foundIf  |
                   ppStatus::finishedIf);
        return;
      }

      bool isTrue;
      if (!lineIsTrue(directive, isTrue)) {
        return;
      }

      pushStatus(ppStatus::foundIf | (isTrue
                                      ? ppStatus::reading
                                      : ppStatus::ignoring));
    }

    void preprocessor_t::processIfdef(identifierToken &directive) {
      // Nested case
      if (status & ppStatus::ignoring) {
        skipToNewline();
        pushStatus(ppStatus::ignoring |
                   ppStatus::foundIf  |
                   ppStatus::finishedIf);
        return;
      }

      bool isTrue;
      if (!getIfdef(directive, isTrue)) {
        return;
      }

      pushStatus(ppStatus::foundIf | (isTrue
                                      ? ppStatus::reading
                                      : ppStatus::ignoring));

      warnOnNonEmptyLine("Extra tokens after macro name");
    }

    void preprocessor_t::processIfndef(identifierToken &directive) {
      // Nested case
      if (status & ppStatus::ignoring) {
        skipToNewline();
        pushStatus(ppStatus::ignoring |
                   ppStatus::foundIf  |
                   ppStatus::finishedIf);
        return;
      }

      bool isTrue;
      if (!getIfdef(directive, isTrue)) {
        return;
      }

      pushStatus(ppStatus::foundIf | (isTrue
                                      ? ppStatus::ignoring
                                      : ppStatus::reading));

      warnOnNonEmptyLine("Extra tokens after macro name");
    }

    void preprocessor_t::processElif(identifierToken &directive) {
      // Check for errors
      if (!(status & ppStatus::foundIf)) {
        errorOn(&directive,
                "#elif without #if");
        skipToNewline();
        return;
      }
      if (status & ppStatus::foundElse) {
        errorOn(&directive,
                "#elif found after an #else directive");
        status &= ~ppStatus::reading;
        status |= (ppStatus::ignoring |
                   ppStatus::finishedIf);
        skipToNewline();
        return;
      }

      // Make sure to test #elif expression is valid
      bool isTrue;
      if (!lineIsTrue(directive, isTrue)) {
        return;
      }

      // If we already finished, keep old state
      if (status & ppStatus::finishedIf) {
        return;
      }

      if (status & ppStatus::reading) {
        swapReadingStatus();
        status |= ppStatus::finishedIf;
      } else if (isTrue) {
        status = (ppStatus::foundIf |
                  ppStatus::reading);
      }
    }

    void preprocessor_t::processElse(identifierToken &directive) {
      warnOnNonEmptyLine("Extra tokens after #else directive");

      // Test errors
      if (!(status & ppStatus::foundIf)) {
        errorOn(&directive,
                "#else without #if");
        return;
      }
      if (status & ppStatus::foundElse) {
        errorOn(&directive,
                "Two #else directives found for the same #if");
        status &= ~ppStatus::reading;
        status |= (ppStatus::ignoring |
                   ppStatus::finishedIf);
        return;
      }

      // Make sure to error on multiple #else
      status |= ppStatus::foundElse;

      // Test status cases
      if (status & ppStatus::finishedIf) {
        return;
      }

      if (status & ppStatus::reading) {
        swapReadingStatus();
        status |= ppStatus::finishedIf;
      } else {
        swapReadingStatus();
      }
    }

    void preprocessor_t::processEndif(identifierToken &directive) {
      warnOnNonEmptyLine("Extra tokens after #endif directive");

      if (!(status & ppStatus::foundIf)) {
        errorOn(&directive,
                "#endif without #if");
      } else {
        popStatus();
      }
    }

    void preprocessor_t::processDefine(identifierToken &directive) {
      token_t *token = getSourceToken();
      if (token_t::safeType(token) != tokenType::identifier) {
        if (!token || passedNewline) {
          incrementNewline();
          errorOn(&directive,
                  "Expected an identifier");
        } else {
          errorOn(token,
                  "Expected an identifier");
        }
        delete token;
        skipToNewline();
        return;
      }

      macro_t &macro = *(new macro_t(*this, token->to<identifierToken>()));
      macro.loadDefinition();

      const std::string &name = macro.name();

      // TODO: Error if the definitions aren't the same
      macroMap::iterator it = sourceMacros.find(name);
      if (it != sourceMacros.end()) {
        delete getMacro(name);
        sourceMacros.erase(it);
      }
      sourceMacros[name] = &macro;

      // Macro clones the token
      delete token;
    }

    void preprocessor_t::processUndef(identifierToken &directive) {
      token_t *token = getSourceToken();
      const int tokenType = token_t::safeType(token);
      if (tokenType != tokenType::identifier) {
        if (tokenType & (tokenType::none |
                         tokenType::newline)) {
          incrementNewline();
          errorOn(&directive,
                  "Expected an identifier");
        } else {
          errorOn(token,
                  "Expected an identifier");
        }
        skipToNewline();
        return;
      }
      // Remove macro
      const std::string &macroName = token->to<identifierToken>().value;
      delete getMacro(macroName);
      macroMap::iterator it = sourceMacros.find(macroName);
      if (it != sourceMacros.end()) {
        sourceMacros.erase(it);
      }
      delete token;
    }

    void preprocessor_t::processError(identifierToken &directive) {
      tokenVector lineTokens;
      getExpandedLineTokens(lineTokens);

      const int tokenCount = (int) lineTokens.size();
      if (!tokenCount) {
        errorOn(&directive,
                "");
      }
      else {
        // Don't include the \n in the message
        const char *start = lineTokens[0]->origin.position.start;
        const char *end   = lineTokens[tokenCount - 1]->origin.position.start;
        const std::string message(start, end - start);
        errorOn(lineTokens[0],
                message);
      }

      freeTokenVector(lineTokens);
    }

    void preprocessor_t::processWarning(identifierToken &directive) {
      tokenVector lineTokens;
      getExpandedLineTokens(lineTokens);

      const int tokenCount = (int) lineTokens.size();
      if (!tokenCount) {
        warningOn(&directive,
                  "");
      }
      else {
        // Don't include the \n in the message
        const char *start = lineTokens[0]->origin.position.start;
        const char *end   = lineTokens[tokenCount - 1]->origin.position.start;
        const std::string message(start, end - start);
        warningOn(lineTokens[0],
                  message);
      }

      freeTokenVector(lineTokens);
    }

    void preprocessor_t::processInclude(identifierToken &directive) {
      // Don't cache since the input might change
      loadTokenizer();
      if (!tokenizer) {
        warningOn(&directive,
                  "Unable to apply #include due to the lack of a tokenizer");
        skipToNewline();
        return;
      }

      // Expand non-absolute path
      bool headerIsQuoted = tokenizer->loadingQuotedHeader();
      std::string header = io::findInPaths(
        io::expandFilename(tokenizer->getHeader(), false),
        includePaths
      );

      if (!io::exists(header)) {
        const bool isSystemHeader =(
          standardHeaders.find(header) != standardHeaders.end()
        );
        const bool includeRawHeader = (
          !strictHeaders || isSystemHeader
        );

        if (strictHeaders && isSystemHeader) {
          warningOn(&directive,
                    "Including standard headers may not be portable for all modes");
        }

        if (includeRawHeader) {
          const std::string line = (
            headerIsQuoted
            ? "include \"" + header + "\""
            : "include <" + header + ">"
          );
          pushOutput(new directiveToken(directive.origin, line));
          return;
        }

        errorOn(&directive,
                "File does not exist");
        skipToNewline();
        return;
      }

      dependencies[header] = true;

      tokenVector lineTokens;
      getExpandedLineTokens(lineTokens);

      if (!header.size()) {
        errorOn(&directive,
                "Expected a header to include");
        freeTokenVector(lineTokens);
        return;
      }

      // Ignore the newline token
      const int lineTokenCount = (int) lineTokens.size();
      if (lineTokenCount > 1) {
        warningOn(lineTokens[0],
                  "Extra tokens after the #include header");
      }
      // In case we read too many tokens, rewind to [\n] token
      if (lineTokenCount) {
        tokenizer->origin = lineTokens[lineTokenCount - 1]->origin;
      }
      freeTokenVector(lineTokens);

      // Clear input cache due to rewind
      while (inputCache.size()) {
        delete inputCache.front();
        inputCache.pop_front();
      }

      // Push source after updating origin to the [\n] token
      input->clearCache();
      tokenizer->pushSource(header);
    }

    void preprocessor_t::processPragma(identifierToken &directive) {
      tokenVector lineTokens;
      getExpandedLineTokens(lineTokens);
      removeNewline(lineTokens);

      const int tokenCount = (int) lineTokens.size();
      if (tokenCount
          && (lineTokens[0]->type() & tokenType::identifier)
          && (((identifierToken*) lineTokens[0])->value == "occa")) {
        processOccaPragma(directive, lineTokens);
        return;
      }

      const std::string value = stringifyTokens(lineTokens, true);

      pushOutput(new pragmaToken(directive.origin,
                                 value));

      freeTokenVector(lineTokens);
    }

    void preprocessor_t::processOccaPragma(identifierToken &directive,
                                           tokenVector &lineTokens) {
      const int tokenCount = (int) lineTokens.size();
      if ((tokenCount < 2)
          || !(lineTokens[1]->type() & tokenType::identifier)
          || (((identifierToken*) lineTokens[1])->value != "attributes")) {
        freeTokenVector(lineTokens);
        return;
      }

      // Insert tokens backwards into input cache, ignoring [occa] [attributes] tokens
      for (int i = (tokenCount - 1); i >= 2; --i) {
        pushInput(lineTokens[i]);
      }
      // Remove the [occa] [attributes] tokens
      delete lineTokens[0];
      delete lineTokens[1];
    }

    void preprocessor_t::processLine(identifierToken &directive) {
      tokenVector lineTokens;
      getExpandedLineTokens(lineTokens);

      // Don't cache since the input might change
      loadTokenizer();
      if (!tokenizer) {
        warningOn(&directive,
                  "Unable to apply #line due to the lack of a tokenizer");
        freeTokenVector(lineTokens);
        return;
      }

      int tokenCount = (int) lineTokens.size();
      if (tokenCount <= 1) {
        token_t *source = (tokenCount
                           ? lineTokens[0]
                           : (token_t*) &directive);
        errorOn(source,
                "Expected a line number");
        freeTokenVector(lineTokens);
        return;
      }

      // Get line number
      int line = -1;
      std::string filename = tokenizer->origin.file->filename;

      token_t *lineToken = lineTokens[0];
      if (lineToken->type() & tokenType::primitive) {
        line = lineToken->to<primitiveToken>().value;
        if (line < 0) {
          errorOn(lineToken,
                  "Line number must be greater or equal to 0");
        }
      } else {
        errorOn(lineToken,
                "Expected a line number");
      }
      if (line < 0) {
        freeTokenVector(lineTokens);
        return;
      }

      // Get new filename
      if (tokenCount > 2) {
        token_t *filenameToken = lineTokens[1];
        if (filenameToken->type() & tokenType::string) {
          filename = filenameToken->to<stringToken>().value;
        } else {
          errorOn(filenameToken,
                  "Expected a filename");
          freeTokenVector(lineTokens);
          return;
        }
      }

      if (tokenCount > 3) {
        warningOn(lineTokens[2],
                  "Extra tokens are unused");
      }

      tokenizer->setOrigin(line, filename);

      freeTokenVector(lineTokens);
    }
    //====================================
  }
}
