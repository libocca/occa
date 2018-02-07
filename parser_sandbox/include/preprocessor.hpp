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

#include "occa/defines.hpp"
#include "occa/types.hpp"

#include "macro.hpp"
#include "trie.hpp"
#include "tokenStream.hpp"

namespace occa {
  namespace lang {
    typedef trie<macro_t*> macroTrie;

    class preprocessor : public tokenStreamTransformWithMap {
    public:
      typedef void (preprocessor::*processDirective_t)(identifierToken &directive);
      typedef trie<processDirective_t> directiveTrie;

      //---[ Status ]-------------------
      std::vector<int> statusStack;
      int currentStatus;

      int expandedIndex;
      tokenVector expandedTokens;

      bool passedNewline;
      //==================================

      //---[ Macros and Directives ]------
      directiveTrie &directives;

      macroTrie compilerMacros;
      macroTrie sourceMacros;
      //==================================

      preprocessor();

      static directiveTrie& getDirectiveTrie();

      void addExpandedToken(token_t *token);

      void pushStatus(const int status);
      int popStatus();

      macro_t* getMacro(const std::string &name);

      virtual token_t* _getToken();
      token_t* getExpandedToken();

      void expandMacro(macro_t &macro);

      void skipToNewline();
      void getTokensUntilNewline(tokenVector &lineTokens);

      token_t* processIdentifier(identifierToken &token);
      token_t* processOperator(operatorToken &token);

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

      template <class TM>
      TM eval(tokenVector &lineTokens) {
        return TM();
      }
    };
  }
}

#endif
