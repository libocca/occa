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
      errorOnToken(NULL),
      directives(getDirectiveTrie()) {

      // Always start off as if we passed a newline
      incrementNewline();

      const int specialMacroCount = 6;
      compilerMacros.autoFreeze = false;
      macro_t *specialMacros[specialMacroCount] = {
        new definedMacro(*this), // defined()
        new fileMacro(*this),    // __FILE__
        new lineMacro(*this),    // __LINE__
        new dateMacro(*this),    // __DATE__
        new timeMacro(*this),    // __TIME__
        new counterMacro(*this)  // __COUNTER__
      };
      for (int i = 0; i < specialMacroCount; ++i) {
        compilerMacros.add(specialMacros[i]->name, specialMacros[i]);
      }

      // Alternative representations
      compilerMacros.add("and"   , new macro_t(*this, "and     &&"));
      compilerMacros.add("and_eq", new macro_t(*this, "and_eq  &="));
      compilerMacros.add("bitand", new macro_t(*this, "bitand  &"));
      compilerMacros.add("bitor" , new macro_t(*this, "bitor   |"));
      compilerMacros.add("compl" , new macro_t(*this, "compl   ~"));
      compilerMacros.add("not"   , new macro_t(*this, "not     !"));
      compilerMacros.add("not_eq", new macro_t(*this, "not_eq  !="));
      compilerMacros.add("or"    , new macro_t(*this, "or      ||"));
      compilerMacros.add("or_eq" , new macro_t(*this, "or_eq   |="));
      compilerMacros.add("xor"   , new macro_t(*this, "xor     ^"));
      compilerMacros.add("xor_eq", new macro_t(*this, "xor_eq  ^="));
      compilerMacros.autoFreeze = true;
      compilerMacros.freeze();

      pushStatus(ppStatus::reading);
    }

    preprocessor::preprocessor(const preprocessor &pp) :
      cacheMap(pp),
      statusStack(pp.statusStack),
      status(pp.status),
      passedNewline(pp.passedNewline),
      errorOnToken(pp.errorOnToken),
      directives(getDirectiveTrie()),
      sourceMacros(pp.sourceMacros) {}

    preprocessor::directiveTrie& preprocessor::getDirectiveTrie() {
      static directiveTrie trie;
      if (trie.isEmpty()) {
        trie.autoFreeze = false;
        trie.add("if"    , &preprocessor::processIf);
        trie.add("ifdef" , &preprocessor::processIfdef);
        trie.add("ifndef", &preprocessor::processIfndef);
        trie.add("elif"  , &preprocessor::processElif);
        trie.add("else"  , &preprocessor::processElse);
        trie.add("endif" , &preprocessor::processEndif);

        trie.add("define", &preprocessor::processDefine);
        trie.add("undef" , &preprocessor::processUndef);

        trie.add("error"  , &preprocessor::processError);
        trie.add("warning", &preprocessor::processWarning);

        trie.add("include", &preprocessor::processInclude);
        trie.add("pragma" , &preprocessor::processPragma);
        trie.add("line"   , &preprocessor::processLine);
        trie.freeze();
      }
      return trie;
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

    tokenMap& preprocessor::cloneMap() const {
      return *(new preprocessor(*this));
    }

    token_t* preprocessor::pop() {
      return getToken();
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

    token_t* preprocessor::getSourceToken() {
      token_t *token;
      *(this->input) >> token;
      return token;
    }

    token_t* preprocessor::getToken() {
      token_t *token = NULL;
      while (!token) {
        decrementNewline();

        token = getSourceToken();
        if (!token) {
          return NULL;
        }
        std::cout << "1. [";
        token->print(std::cout);
        std::cout << "] ("
                  << ((status & ppStatus::ignoring)
                      ? "ignoring"
                      : "reading")
                  << ")\n";

        const int tokenType = token->type();
        if (tokenType & tokenType::identifier) {
          token = processIdentifier(token->to<identifierToken>());
        } else if (tokenType & tokenType::op) {
          token = processOperator(token->to<operatorToken>());
        } else if (tokenType & tokenType::newline) {
          incrementNewline();
          push(token);
          token = NULL;
        }
        std::cout << "2. [";
        if (token) {
          token->print(std::cout);
        }
        std::cout << "] ("
                  << ((status & ppStatus::ignoring)
                      ? "ignoring"
                      : "reading")
                  << ")\n";

        // Ignore tokens inside disabled #if/#elif/#else regions
        if (status & ppStatus::ignoring) {
          delete token;
          token = NULL;
        }
      }
      return token;
    }

    void preprocessor::expandMacro(macro_t &macro) {
      // TODO
    }

    void preprocessor::skipToNewline() {
      token_t *token = getSourceToken();
      while (token) {
        const int tokenType = token->type();
        if (tokenType & tokenType::newline) {
          incrementNewline();
          push(token);
          return;
        }
        delete token;
        token = getSourceToken();
      }
    }

    void preprocessor::getTokensUntilNewline(tokenVector &lineTokens) {
      while (true) {
        token_t *token = getSourceToken();
        if (!token) {
          break;
        }
        if (token->type() & tokenType::newline) {
          incrementNewline();
          // Add the newline token back to the queue
          // Only used by processIf, so nothing else
          //   should be expanding in between...
          push(token);
          break;
        }
        lineTokens.push_back(token);
      }
    }

    token_t* preprocessor::processIdentifier(identifierToken &token) {
      macro_t *macro = getMacro(token.value);
      if (macro) {
        if (!macro->isFunctionLike()) {
          expandMacro(*macro);
          delete &token;
          return NULL;
        }
        // Make sure that the macro starts with a '('
        token_t *nextToken = getSourceToken();
        if (token_t::safeType(nextToken) & tokenType::op) {
          const opType_t opType = nextToken->to<operatorToken>().op.opType;
          if (opType & operatorType::parenthesesEnd) {
            expandMacro(*macro);
            delete &token;
            delete nextToken;
            return NULL;
          }
        }
        push(nextToken);
      }
      return &token;
    }

    token_t* preprocessor::processOperator(operatorToken &token) {
      if ((token.op.opType != operatorType::hash) ||
          !passedNewline) {
        return &token;
      }
      delete &token;

      // Find directive
      token_t *directive = getSourceToken();
      // NULL or an empty # is ok
      if (!directive ||
          (directive->type() & tokenType::newline)) {
        incrementNewline();
        return directive;
      }

      // Check for valid directive
      if (directive->type() != tokenType::identifier) {
        errorOn(directive,
                "Unknown preprocessor directive");
        skipToNewline();
        return NULL;
      }
      identifierToken &directiveToken = directive->to<identifierToken>();
      const std::string &directiveStr = directiveToken.value;
      directiveTrie::result_t result  = directives.get(directiveStr);
      if (!result.success()) {
        errorOn(directive,
                "Unknown preprocessor directive");
        delete directive;
        skipToNewline();
        return NULL;
      }
      (this->*(result.value()))(directiveToken);
      delete directive;
      return NULL;
    }

    void preprocessor::processIf(identifierToken &directive) {
      tokenVector lineTokens;
      getTokensUntilNewline(lineTokens);
      if (!lineTokens.size()) {
        errorOn(&directive,
                "Expected a value or expression");
        // Default to #if false
        pushStatus(ppStatus::ignoring |
                   ppStatus::foundIf);
        return;
      }
      pushStatus(ppStatus::foundIf | (eval<bool>(lineTokens)
                                      ? ppStatus::reading
                                      : ppStatus::ignoring));
    }

    void preprocessor::processIfdef(identifierToken &directive) {
      token_t *token = getSourceToken();
      const int tokenType = token_t::safeType(token);

      if (!(tokenType & tokenType::identifier)) {
        // Print from the directive if we don't
        //   have a token in the same line
        if (tokenType & (tokenType::none |
                         tokenType::newline)) {
          incrementNewline();
          errorOn(&directive,
                  "Expected an identifier");
        } else {
          errorOn(token,
                  "Expected an identifier");
          skipToNewline();
        }
        // Default to false
        pushStatus(ppStatus::foundIf |
                   ppStatus::ignoring);
        return;
      }

      const std::string &macroName = token->to<identifierToken>().value;
      delete token;
      pushStatus(ppStatus::foundIf | (getMacro(macroName)
                                      ? ppStatus::reading
                                      : ppStatus::ignoring));
      skipToNewline();
    }

    void preprocessor::processIfndef(identifierToken &directive) {
      const int oldErrors = errors;
      processIfdef(directive);
      // Keep the ignoring status if ifdef found an error
      if (oldErrors == errors) {
        if (status & ppStatus::reading) {
          status &= ~ppStatus::reading;
          status |= ppStatus::ignoring;
        } else {
          status &= ~ppStatus::ignoring;
          status |= ppStatus::reading;
        }
      }
    }

    void preprocessor::processElif(identifierToken &directive) {
      if (!(status & ppStatus::foundIf)) {
        errorOn(&directive,
                "#elif without #if");
        skipToNewline();
      }
      else if (status & ppStatus::foundElse) {
        errorOn(&directive,
                "#elif found after an #else directive");
        status &= ~ppStatus::reading;
        status |= (ppStatus::ignoring |
                   ppStatus::finishedIf);
        skipToNewline();
      }
      else if (status & ppStatus::finishedIf) {
        skipToNewline();
      }
      else {
        // processIf pushes status, we only want to
        //   update it
        processIf(directive);
        status = popStatus();
      }
    }

    void preprocessor::processElse(identifierToken &directive) {
      if (!(status & ppStatus::foundIf)) {
        errorOn(&directive,
                "#else without #if");
      }
      else if (status & ppStatus::foundElse) {
        errorOn(&directive,
                "Two #else directives found for the same #if");
        status &= ~ppStatus::reading;
        status |= (ppStatus::ignoring |
                   ppStatus::finishedIf);
      }
      else if (!(status & ppStatus::finishedIf)) {
        status &= ~ppStatus::ignoring;
        status |= (ppStatus::reading |
                   ppStatus::finishedIf);
      }
      status |= ppStatus::foundElse;
      skipToNewline();
    }

    void preprocessor::processEndif(identifierToken &directive) {
      if (!(status & ppStatus::foundIf)) {
        errorOn(&directive,
                "#endif without #if");
      }
      // TODO: Test if we're inside an if
      if (!popStatus()) {
        printError("#endif without #if");
      }
      skipToNewline();
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
        skipToNewline();
        return;
      }
      // TODO
      // macro_t *macro;
      // sourceMacros.add(macro->name, macro);
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
      // TODO
      const std::string message = "message";
      errorOn(&directive,
              message);
      skipToNewline();
    }

    void preprocessor::processWarning(identifierToken &directive) {
      // TODO
      const std::string message = "message";
      directive.printWarning(message);
      skipToNewline();
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
