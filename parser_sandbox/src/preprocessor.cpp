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
#include <sstream>
#include <stdlib.h>

#include "occa/tools/hash.hpp"
#include "occa/tools/io.hpp"
#include "occa/tools/lex.hpp"
#include "occa/tools/string.hpp"
#include "occa/parser/primitive.hpp"

#include "preprocessor.hpp"
#include "specialMacros.hpp"
#include "expression.hpp"

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
    preprocessor::preprocessor() :
      errorOnToken(NULL) {

      // Always start off as if we passed a newline
      incrementNewline();

      initDirectives();

      compilerMacros.autoFreeze = false;

      const int specialMacroCount = 7;
      macro_t *specialMacros[specialMacroCount] = {
        new definedMacro(*this),    // defined()
        new hasIncludeMacro(*this), // __has_include()
        new fileMacro(*this),       // __FILE__
        new lineMacro(*this),       // __LINE__
        new dateMacro(*this),       // __DATE__
        new timeMacro(*this),       // __TIME__
        new counterMacro(*this)     // __COUNTER__
      };
      for (int i = 0; i < specialMacroCount; ++i) {
        compilerMacros.add(specialMacros[i]->name(), specialMacros[i]);
      }

      // Alternative representations
      compilerMacros.add("and"   , macro_t::defineBuiltin(*this, "and"   , "&&"));
      compilerMacros.add("and_eq", macro_t::defineBuiltin(*this, "and_eq", "&="));
      compilerMacros.add("bitand", macro_t::defineBuiltin(*this, "bitand", "&"));
      compilerMacros.add("bitor" , macro_t::defineBuiltin(*this, "bitor" , "|"));
      compilerMacros.add("compl" , macro_t::defineBuiltin(*this, "compl" , "~"));
      compilerMacros.add("not"   , macro_t::defineBuiltin(*this, "not"   , "!"));
      compilerMacros.add("not_eq", macro_t::defineBuiltin(*this, "not_eq", "!="));
      compilerMacros.add("or"    , macro_t::defineBuiltin(*this, "or"    , "||"));
      compilerMacros.add("or_eq" , macro_t::defineBuiltin(*this, "or_eq" , "|="));
      compilerMacros.add("xor"   , macro_t::defineBuiltin(*this, "xor"   , "^"));
      compilerMacros.add("xor_eq", macro_t::defineBuiltin(*this, "xor_eq", "^="));

      compilerMacros.freeze();
      compilerMacros.autoFreeze = true;

      pushStatus(ppStatus::reading);
    }

    preprocessor::preprocessor(const preprocessor &pp) :
      cacheMap(pp) {
      *this = pp;
    }

    preprocessor::~preprocessor() {
      macroTrie *tries[2] = {
        &compilerMacros,
        &sourceMacros
      };

      for (int i = 0; i < 2; ++i) {
        macroTrie &trie = *(tries[i]);
        const int trieSize = trie.size();
        for (int j = 0; j < trieSize; ++j) {
          delete trie.values[j];
        }
      }
    }

    preprocessor& preprocessor::operator = (const preprocessor &pp) {
      statusStack   = pp.statusStack;
      status        = pp.status;
      passedNewline = pp.passedNewline;
      errorOnToken  = pp.errorOnToken;

      directives     = pp.directives;
      compilerMacros = pp.compilerMacros;
      sourceMacros   = pp.sourceMacros;

      directives.freeze();
      directives.autoFreeze = true;

      macroTrie *tries[2] = {
        &compilerMacros,
        &sourceMacros
      };

      for (int i = 0; i < 2; ++i) {
        macroTrie &trie = *(tries[i]);
        const int trieSize = trie.size();
        for (int j = 0; j < trieSize; ++j) {
          trie.values[j] = &(trie.values[j]->clone());
        }
      }

      compilerMacros.autoFreeze = true;
      sourceMacros.autoFreeze   = true;

      return *this;
    }

    void preprocessor::initDirectives() {
      directives.autoFreeze = false;
      directives.add("if"     , &preprocessor::processIf);
      directives.add("ifdef"  , &preprocessor::processIfdef);
      directives.add("ifndef" , &preprocessor::processIfndef);
      directives.add("elif"   , &preprocessor::processElif);
      directives.add("else"   , &preprocessor::processElse);
      directives.add("endif"  , &preprocessor::processEndif);

      directives.add("define" , &preprocessor::processDefine);
      directives.add("undef"  , &preprocessor::processUndef);

      directives.add("error"  , &preprocessor::processError);
      directives.add("warning", &preprocessor::processWarning);

      directives.add("include", &preprocessor::processInclude);
      directives.add("pragma" , &preprocessor::processPragma);
      directives.add("line"   , &preprocessor::processLine);
      directives.freeze();
      directives.autoFreeze = true;
    }

    void preprocessor::preprint(std::ostream &out) {
      errorOnToken->preprint(out);
    }

    void preprocessor::postprint(std::ostream &out) {
      errorOnToken->postprint(out);
    }

    void preprocessor::errorOn(token_t *token,
                               const std::string &message) {
      errorOnToken = token;
      if (token) {
        printError(message);
      }
      errorOnToken = NULL;
    }

    tokenMap& preprocessor::clone_() const {
      return *(new preprocessor(*this));
    }

    void preprocessor::pushStatus(const int status_) {
      statusStack.push_back(status);
      status = status_;
    }

    int preprocessor::popStatus() {
      if (statusStack.size() == 0) {
        return 0;
      }
      status = statusStack.back();
      statusStack.pop_back();
      return status;
    }

    void preprocessor::swapReadingStatus() {
      if (status & ppStatus::reading) {
        status &= ~ppStatus::reading;
        status |= ppStatus::ignoring;
      } else {
        status &= ~ppStatus::ignoring;
        status |= ppStatus::reading;
      }
    }

    void preprocessor::incrementNewline() {
      // We need to keep passedNewline 'truthy'
      //   until after the next token
      passedNewline = 2;
    }

    void preprocessor::decrementNewline() {
      passedNewline -= !!passedNewline;
    }

    macro_t* preprocessor::getMacro(const std::string &name) {
      macroTrie::result_t result = sourceMacros.get(name);
      if (result.success()) {
        return result.value();
      }
      result = compilerMacros.get(name);
      if (result.success()) {
        return result.value();
      }
      return NULL;
    }

    bool preprocessor::hasSourceTokens() {
      return (!inputIsEmpty() ||
              !sourceCache.empty());
    }

    token_t* preprocessor::getSourceToken() {
      token_t *token = NULL;

      if (sourceCache.size()) {
        token = sourceCache.front();
        sourceCache.pop_front();
        return token;
      }

      if (!inputIsEmpty()) {
        *(input) >> token;
      }
      return token;
    }

    bool preprocessor::isEmpty() {
      if (!cache.empty()) {
        return false;
      }

      while (cache.empty() &&
             hasSourceTokens()) {
        pop();
      }

      return cache.empty();
    }

    void preprocessor::pop() {
      processToken(getSourceToken());
    }

    void preprocessor::expandMacro(identifierToken &source,
                                   macro_t &macro) {
      tokenVector tokens;
      macro.expand(tokens, source);

      // Push tokens to the front backwards to
      //   maintain the same order
      const int tokenCount = (int) tokens.size();
      for (int i = (tokenCount - 1); i >= 0; --i) {
        sourceCache.push_front(tokens[i]);
      }
    }

    void preprocessor::skipToNewline() {
      tokenVector lineTokens;
      getLineTokens(lineTokens);

      // Push the newline token
      const int tokens = (int) lineTokens.size();
      if (tokens) {
        push(lineTokens[tokens-1]);
        lineTokens.pop_back();
      }

      freeTokenVector(lineTokens);
    }

    // lineTokens might be partially initialized
    //   so we don't want to clear it
    void preprocessor::getLineTokens(tokenVector &lineTokens) {
      while (hasSourceTokens()) {
        token_t *token = getSourceToken();

        if (token->type() & tokenType::newline) {
          incrementNewline();
          lineTokens.push_back(token);
          break;
        }

        lineTokens.push_back(token);
      }
    }

    void preprocessor::warnOnNonEmptyLine(const std::string &message) {
      tokenVector lineTokens;
      getLineTokens(lineTokens);
      if (lineTokens.size()) {
        // Don't account for the newline token
        if (lineTokens[0]->type() != tokenType::newline) {
          lineTokens[0]->printWarning(message);
        }
        freeTokenVector(lineTokens);
      }
    }

    void preprocessor::processToken(token_t *token) {
      decrementNewline();

      const int tokenType = token->type();

      if (tokenType & tokenType::newline) {
        incrementNewline();
        push(token);
        return;
      }

      // Only process operators when ignoring
      //   for potential #
      if (status & ppStatus::ignoring) {
        if (!(tokenType & tokenType::op)) {
          delete token;
          return;
        }
        const operator_t &op = token->to<operatorToken>().op;
        if (!(op.opType & operatorType::preprocessor)) {
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
        push(token);
      }
    }

    void preprocessor::processIdentifier(identifierToken &token) {
      // Ignore tokens inside disabled #if/#elif/#else regions
      if (status & ppStatus::ignoring) {
        delete &token;
        return;
      }

      macro_t *macro = getMacro(token.value);
      if (macro) {
        if (!macro->isFunctionLike()) {
          expandMacro(token, *macro);
          delete &token;
          return;
        }

        if (!hasSourceTokens()) {
          push(&token);
          return;
        }

        // Make sure that the macro starts with a '('
        token_t *nextToken = getSourceToken();
        if (token_t::safeType(nextToken) & tokenType::op) {
          const opType_t opType = nextToken->to<operatorToken>().op.opType;
          if (opType & operatorType::parenthesesStart) {
            expandMacro(token, *macro);
            delete &token;
            delete nextToken;
            return;
          }
        }
        // Prioritize possible variable if no () is found:
        //   #define FOO()
        //   int FOO;
        if (nextToken) {
          sourceCache.push_front(&token);
          sourceCache.push_front(nextToken);
        } else {
          push(&token);
        }
        return;
      }

      push(&token);
    }

    void preprocessor::processOperator(operatorToken &token) {
      if ((token.op.opType != operatorType::hash) ||
          !passedNewline) {
        push(&token);
        return;
      }
      delete &token;

      if (!hasSourceTokens()) {
        return;
      }

      // NULL or an empty # is ok
      token_t *directive = getSourceToken();
      if (directive->type() & tokenType::newline) {
        incrementNewline();
        push(directive);
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
      directiveTrie::result_t result  = directives.get(directiveStr);
      if (!result.success()) {
        errorOn(directive,
                "Unknown preprocessor directive");
        delete directive;
        skipToNewline();
        return;
      }

      (this->*(result.value()))(directiveToken);

      delete directive;
    }


    bool preprocessor::lineIsTrue(identifierToken &directive) {
      tokenVector lineTokens;
      getLineTokens(lineTokens);

      // Remove the newline token
      if (lineTokens.size()) {
        delete lineTokens.back();
        lineTokens.pop_back();
      }

      exprNode *expr = getExpression(lineTokens);

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

      freeTokenVector(lineTokens);

      // Default to #if false with error
      bool isTrue = false;
      if (exprError) {
        pushStatus(ppStatus::ignoring |
                   ppStatus::foundIf);
      } else {
        isTrue = expr->evaluate();
      }

      delete expr;
      return isTrue;
    }

    void preprocessor::processIf(identifierToken &directive) {
      // Nested case
      if (status & ppStatus::ignoring) {
        skipToNewline();
        pushStatus(ppStatus::ignoring |
                   ppStatus::foundIf  |
                   ppStatus::finishedIf);
        return;
      }

      const bool isTrue = lineIsTrue(directive);

      pushStatus(ppStatus::foundIf | (isTrue
                                      ? ppStatus::reading
                                      : ppStatus::ignoring));
    }

    void preprocessor::processIfdef(identifierToken &directive) {
      // Nested case
      if (status & ppStatus::ignoring) {
        skipToNewline();
        pushStatus(ppStatus::ignoring |
                   ppStatus::foundIf  |
                   ppStatus::finishedIf);
        return;
      }

      token_t *token = getSourceToken();
      const int tokenType = token_t::safeType(token);

      if (!(tokenType & tokenType::identifier)) {
          // Print from the directive if we don't
          //   have a token in the same line
        token_t *errorToken = &directive;
        if (tokenType & tokenType::newline) {
          incrementNewline();
          push(token);
        } else if (tokenType & ~tokenType::none) {
          errorToken = token;
        }
        errorOn(errorToken,
                "Expected an identifier");

        // Default to false
        pushStatus(ppStatus::ignoring |
                   ppStatus::foundIf);
        return;
      }

      const std::string &macroName = token->to<identifierToken>().value;
      pushStatus(ppStatus::foundIf | (getMacro(macroName)
                                      ? ppStatus::reading
                                      : ppStatus::ignoring));

      warnOnNonEmptyLine("Extra tokens after macro name");
      delete token;
    }

    void preprocessor::processIfndef(identifierToken &directive) {
      const int oldStatus = status;
      const int oldErrors = errors;
      processIfdef(directive);
      // Keep the ignoring status if ifdef found an error
      if (oldErrors != errors) {
        return;
      }
      // If we're in a nested #if 0, keep the status
      if (oldStatus & ppStatus::ignoring) {
        return;
      }
      swapReadingStatus();
    }

    void preprocessor::processElif(identifierToken &directive) {
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
      const bool isTrue = lineIsTrue(directive);

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

    void preprocessor::processElse(identifierToken &directive) {
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

    void preprocessor::processEndif(identifierToken &directive) {
      warnOnNonEmptyLine("Extra tokens after #endif directive");

      if (!(status & ppStatus::foundIf)) {
        errorOn(&directive,
                "#endif without #if");
      } else {
        popStatus();
      }
    }

    void preprocessor::processDefine(identifierToken &directive) {
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
      sourceMacros.add(macro.name(), &macro);

      // Macro clones the token
      delete token;
    }

    void preprocessor::processUndef(identifierToken &directive) {
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
      delete token;
      sourceMacros.remove(macroName);
    }

    void preprocessor::processError(identifierToken &directive) {
      tokenVector lineTokens;
      getLineTokens(lineTokens);

      const int tokenCount = (int) lineTokens.size();
      if (!tokenCount) {
        directive.printError("");
      }
      else {
        // Don't include the \n in the message
        const char *start = lineTokens[0]->origin.position.start;
        const char *end   = lineTokens[tokenCount - 1]->origin.position.start;
        const std::string message(start, end - start);
        lineTokens[0]->printError(message);
      }

      freeTokenVector(lineTokens);
    }

    void preprocessor::processWarning(identifierToken &directive) {
      tokenVector lineTokens;
      getLineTokens(lineTokens);

      const int tokenCount = (int) lineTokens.size();
      if (!tokenCount) {
        directive.printWarning("");
      }
      else {
        // Don't include the \n in the message
        const char *start = lineTokens[0]->origin.position.start;
        const char *end   = lineTokens[tokenCount - 1]->origin.position.start;
        const std::string message(start, end - start);
        lineTokens[0]->printWarning(message);
      }

      freeTokenVector(lineTokens);
    }

    void preprocessor::processInclude(identifierToken &directive) {
      // TODO
    }

    void preprocessor::processPragma(identifierToken &directive) {
      // TODO
    }

    void preprocessor::processLine(identifierToken &directive) {
      // TODO
    }
    //====================================
  }
}
