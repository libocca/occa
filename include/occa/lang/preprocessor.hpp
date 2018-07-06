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
#ifndef OCCA_LANG_PREPROCESSOR_HEADER
#define OCCA_LANG_PREPROCESSOR_HEADER

#include <ostream>
#include <vector>
#include <map>
#include <stack>

#include <occa/defines.hpp>
#include <occa/types.hpp>
#include <occa/lang/macro.hpp>
#include <occa/lang/token.hpp>
#include <occa/lang/stream.hpp>

namespace occa {
  namespace lang {
    class tokenizer_t;

    typedef std::vector<token_t*> tokenVector;
    typedef std::stack<token_t*>  tokenStack;
    typedef std::list<token_t*>   tokenList;

    typedef std::map<std::string, macro_t*> macroMap;
    typedef std::map<macro_t*, bool>        macroSet;
    typedef std::vector<macro_t*>           macroVector;
    typedef std::map<token_t*, macroVector> macroEndMap;

    typedef streamMap<token_t*, token_t*> tokenMap;

    namespace ppStatus {
      extern const int reading;
      extern const int ignoring;
      extern const int foundIf;
      extern const int foundElse;
      extern const int finishedIf;
    }

    class preprocessor_t : public withInputCache<token_t*, token_t*>,
                           public withOutputCache<token_t*, token_t*> {
    public:
      typedef void (preprocessor_t::*processDirective_t)(identifierToken &directive);
      typedef std::map<std::string, processDirective_t> directiveMap;

      //---[ Status ]-------------------
      std::vector<int> statusStack;
      int status;

      int passedNewline;
      bool expandingMacros;

      macroSet expandedMacros;
      macroEndMap expandedMacroEnd;
      //================================

      //---[ Macros and Directives ]----
      directiveMap directives;

      macroMap compilerMacros;
      macroMap sourceMacros;
      //================================

      //---[ Metadata ]-----------------
      strToBoolMap dependencies;
      int warnings, errors;
      //================================

      //---[ Misc ]---------------------
      strVector includePaths;
      //================================

      preprocessor_t(const occa::properties &settings_ = occa::properties());
      preprocessor_t(const preprocessor_t &other);
      ~preprocessor_t();

      void init();
      void clear();
      void clear_();

      preprocessor_t& operator = (const preprocessor_t &pp);

      void initDirectives();

      void warningOn(token_t *token,
                     const std::string &message);

      void errorOn(token_t *token,
                   const std::string &message);

      virtual tokenMap& clone_() const;

      virtual void* passMessageToInput(const occa::properties &props);

      void pushStatus(const int status_);
      int popStatus();
      void swapReadingStatus();

      void incrementNewline();
      void decrementNewline();

      macro_t* getMacro(const std::string &name);
      macro_t* getSourceMacro();

      token_t* getSourceToken();

      virtual void fetchNext();

      //---[ Public ]-------------------
      void addCompilerDefine(const std::string &name,
                             const std::string &value);

      void removeCompilerDefine(const std::string &name);

      void addSourceDefine(const std::string &name,
                           const std::string &value);

      void removeSourceDefine(const std::string &name);
      //================================

      void expandMacro(identifierToken &source,
                       macro_t &macro);
      void clearExpandedMacros(token_t *token);

      void skipToNewline();
      void getLineTokens(tokenVector &lineTokens);
      void getExpandedLineTokens(tokenVector &lineTokens);
      void warnOnNonEmptyLine(const std::string &message);
      void removeNewline(tokenVector &lineTokens);

      void processToken(token_t *token);
      void processIdentifier(identifierToken &token);
      void processOperator(operatorToken &token);

      bool lineIsTrue(identifierToken &directive,
                      bool &isTrue);
      bool getIfdef(identifierToken &directive,
                    bool &isTrue);

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
      void processOccaPragma(identifierToken &directive,
                             tokenVector &lineTokens);

      int getLineNumber();
      void processLine(identifierToken &directive);
    };
  }
}

#endif
