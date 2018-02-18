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
#ifndef OCCA_PARSER_PREPROCESSOR_HEADER2
#define OCCA_PARSER_PREPROCESSOR_HEADER2

#include <ostream>
#include <vector>
#include <map>
#include <deque>

#include "occa/defines.hpp"
#include "occa/types.hpp"

#include "macro.hpp"
#include "trie.hpp"
#include "stream.hpp"
#include "token.hpp"

namespace occa {
  namespace lang {
    typedef trie<macro_t*> macroTrie;
    typedef streamMap<token_t*, token_t*> tokenMap;
    typedef cacheMap<token_t*, token_t*> tokenCacheMap;
    typedef std::deque<token_t*> tokenDeque;

    namespace ppStatus {
      extern const int reading;
      extern const int ignoring;
      extern const int foundIf;
      extern const int foundElse;
      extern const int finishedIf;
    }

    class preprocessor : public tokenCacheMap,
                         public errorHandler {
    public:
      typedef void (preprocessor::*processDirective_t)(identifierToken &directive);
      typedef trie<processDirective_t> directiveTrie;

      tokenDeque sourceCache;

      //---[ Status ]-------------------
      std::vector<int> statusStack;
      int status;

      int passedNewline;
      token_t *errorOnToken;
      //==================================

      //---[ Macros and Directives ]------
      directiveTrie directives;

      macroTrie compilerMacros;
      macroTrie sourceMacros;
      //==================================

      preprocessor();
      preprocessor(const preprocessor &pp);
      ~preprocessor();

      preprocessor& operator = (const preprocessor &pp);

      virtual void preprint(std::ostream &out);

      virtual void postprint(std::ostream &out);

      void errorOn(token_t *token,
                   const std::string &message);

      virtual tokenMap& clone_() const;

      void initDirectives();

      void addExpandedToken(token_t *token);

      void pushStatus(const int status_);
      int popStatus();
      void swapReadingStatus();

      void incrementNewline();
      void decrementNewline();

      macro_t* getMacro(const std::string &name);
      macro_t* getSourceMacro();

      bool hasSourceTokens();
      token_t* getSourceToken();

      virtual bool isEmpty();
      virtual void pop();

      void expandMacro(identifierToken &source,
                       macro_t &macro);

      void skipToNewline();
      void getLineTokens(tokenVector &lineTokens);
      void warnOnNonEmptyLine(const std::string &message);

      void processToken(token_t *token);
      void processIdentifier(identifierToken &token);
      void processOperator(operatorToken &token);

      bool lineIsTrue(identifierToken &directive);
      void processIf(identifierToken &directive);
      void processIfdef(identifierToken &directive);
      void processIfndef(identifierToken &directive);
      void processElif(identifierToken &directive);
      void processElse(identifierToken &directive);
      void processEndif(identifierToken &directive);

      void processDefine(identifierToken &directive);
      void processUndef(identifierToken &directive);

      void processError(identifierToken &directive);
      void processWarning(identifierToken &directive);

      void processInclude(identifierToken &directive);
      void processPragma(identifierToken &directive);
      void processLine(identifierToken &directive);
    };
  }
}

#endif
