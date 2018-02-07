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
    //|-----[ Status ]--------------------
    static const int reading    = (1 << 0);
    static const int ignoring   = (1 << 1);
    static const int foundElse  = (1 << 2);
    static const int finishedIf = (1 << 3);
    //|===================================

    // TODO: Add actual compiler macros as well
    preprocessor::preprocessor() :
      expandedIndex(0),
      passedNewline(true),
      directives(getDirectiveTrie()) {

      compilerMacros.autoFreeze = false;
      macro_t *specialMacros[5] = {
        new fileMacro(this),   // __FILE__
        new lineMacro(this),   // __LINE__
        new dateMacro(this),   // __DATE__
        new timeMacro(this),   // __TIME__
        new counterMacro(this) // __COUNTER__
      };
      for (int i = 0; i < 5; ++i) {
        compilerMacros.add(specialMacros[i]->name, specialMacros[i]);
      }

      // Alternative representations
      compilerMacros.add("and"   , new macro_t(this, "and     &&"));
      compilerMacros.add("and_eq", new macro_t(this, "and_eq  &="));
      compilerMacros.add("bitand", new macro_t(this, "bitand  &"));
      compilerMacros.add("bitor" , new macro_t(this, "bitor   |"));
      compilerMacros.add("compl" , new macro_t(this, "compl   ~"));
      compilerMacros.add("not"   , new macro_t(this, "not     !"));
      compilerMacros.add("not_eq", new macro_t(this, "not_eq  !="));
      compilerMacros.add("or"    , new macro_t(this, "or      ||"));
      compilerMacros.add("or_eq" , new macro_t(this, "or_eq   |="));
      compilerMacros.add("xor"   , new macro_t(this, "xor     ^"));
      compilerMacros.add("xor_eq", new macro_t(this, "xor_eq  ^="));

      pushStatus(reading);
    }

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

    void preprocessor::addExpandedToken(token_t *token) {
      expandedTokens.push_back(token);
    }

    void preprocessor::pushStatus(const int status) {
      statusStack.push_back(currentStatus);
      currentStatus = status;
    }

    int preprocessor::popStatus() {
      if (!statusStack.size()) {
        return 0;
      }
      currentStatus = statusStack.back();
      statusStack.pop_back();
      return currentStatus;
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

    token_t* preprocessor::_getToken() {
      passedNewline = false;

      // Check if we've expanded tokens
      token_t *token = getExpandedToken();
      if (token) {
        return token;
      }
      while (!token) {
        // Get next token
        token = getSourceToken();
        if (!token) {
          return NULL;
        }
        const int tokenType = token->type();
        if (tokenType & tokenType::identifier) {
          token = processIdentifier(token->to<identifierToken>());
        } else if (tokenType & tokenType::op) {
          token = processOperator(token->to<operatorToken>());
        } else if (tokenType & tokenType::newline) {
          passedNewline = true;
        }
      }
      return token;
    }

    token_t* preprocessor::getExpandedToken() {
      const int expandCount = expandedTokens.size();
      if (expandCount) {
        if (expandedIndex < expandCount) {
          token_t *token = expandedTokens[expandedIndex++];
          if (token_t::safeType(token) & tokenType::newline) {
            passedNewline = true;
          }
          return token;
        }
        expandedIndex = 0;
        expandedTokens.clear();
      }
      return NULL;
    }

    void preprocessor::expandMacro(macro_t &macro) {
      // TODO
    }

    void preprocessor::skipToNewline() {
      token_t *token = _getToken();
      while (token) {
        const int tokenType = token->type();
        delete token;
        if (tokenType & tokenType::newline) {
          return;
        }
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
          delete token;
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
          return getExpandedToken();
        }
        // Make sure that the macro starts with a '('
        token_t *nextToken = getSourceToken();
        if (token_t::safeType(nextToken) & tokenType::op) {
          const opType_t opType = nextToken->to<operatorToken>().op.opType;
          if (opType & operatorType::parenthesesEnd) {
            expandMacro(*macro);
            delete &token;
            delete nextToken;
            return getExpandedToken();
          }
        }
        expandedTokens.push_back(nextToken);
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
      if (!directive || passedNewline) {
        return directive;
      }

      // Check for valid directive
      if (directive->type() != tokenType::identifier) {
        directive->printError("Unknown preprocessor directive");
        skipToNewline();
        return NULL;
      }
      identifierToken &directiveToken = directive->to<identifierToken>();
      const std::string &directiveStr = directiveToken.value;
      directiveTrie::result_t result  = directives.get(directiveStr);
      if (!result.success()) {
        directive->printError("Unknown preprocessor directive");
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
        directive.printError("Expected a value");
        pushStatus(ignoring); // Keep checking for errors
        return;
      }
      pushStatus(eval<bool>(lineTokens)
                 ? reading
                 : ignoring);
    }

    void preprocessor::processIfdef(identifierToken &directive) {
      token_t *token = getSourceToken();
      if (!token
          || (token->type() != tokenType::identifier)
          || passedNewline) {
        if (!token || passedNewline) {
          directive.printError("Expected an identifier");
        } else {
          token->printError("Expected an identifier");
        }
        pushStatus(ignoring);
        return;
      }
      const std::string &macroName = token->to<identifierToken>().value;
      delete token;
      pushStatus(getMacro(macroName)
                 ? reading
                 : ignoring);
      skipToNewline();
    }

    void preprocessor::processIfndef(identifierToken &directive) {
      const int oldErrors = errors;
      processIfdef(directive);
      // Keep the ignoring status if ifdef found an error
      if (oldErrors == errors) {
        const int status = ((currentStatus == reading)
                            ? ignoring
                            : reading);
        popStatus();
        pushStatus(status);
      }
    }

    void preprocessor::processElif(identifierToken &directive) {
      // TODO: Test if we're inside an if
      processIf(directive);
    }

    void preprocessor::processElse(identifierToken &directive) {
      // TODO: Test if we're inside an if
      skipToNewline();
    }

    void preprocessor::processEndif(identifierToken &directive) {
      // TODO: Test if we're inside an if
      if (!popStatus()) {
        printError("#endif without #if");
      }
      skipToNewline();
    }

    void preprocessor::processDefine(identifierToken &directive) {
      token_t *token = getSourceToken();
      if (!token
          || (token->type() != tokenType::identifier)
          || passedNewline) {
        if (!token || passedNewline) {
          directive.printError("Expected an identifier");
        } else {
          token->printError("Expected an identifier");
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
      if (token->type() != tokenType::identifier) {
        if (tokenType & (tokenType::none |
                         tokenType::newline)) {
          directive.printError("Expected an identifier");
        } else {
          token->printError("Expected an identifier");
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
      directive.printError(message);
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
