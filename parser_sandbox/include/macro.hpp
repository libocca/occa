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
#ifndef OCCA_PARSER_MACRO_HEADER2
#define OCCA_PARSER_MACRO_HEADER2

#include <vector>
#include "token.hpp"
#include "trie.hpp"

namespace occa {
  namespace lang {
    class token_t;
    class identifierToken;
    class macroToken;
    class preprocessor;

    typedef trie<int> intTrie;

    typedef std::vector<token_t*>    tokenVector;
    typedef std::vector<macroToken*> macroTokenVector_t;

    //---[ Macro Tokens ]---------------
    class macroToken {
    public:
      token_t *thisToken;

      macroToken(token_t *thisToken_);
      virtual ~macroToken();

      virtual bool expand(tokenVector &newTokens,
                          token_t *source,
                          std::vector<tokenVector> &args) = 0;

      std::string stringifyTokens(tokenVector &tokens,
                                  const bool addSpaces);
    };

    class macroRawToken : public macroToken {
    public:
      macroRawToken(token_t *token_);

      virtual bool expand(tokenVector &newTokens,
                          token_t *source,
                          std::vector<tokenVector> &args);
    };

    class macroArgument : public macroToken {
    public:
      int arg;
      int argc;

      macroArgument(token_t *token_,
                    const int arg_,
                    const int argc_);

      void expandArg(tokenVector &newTokens,
                     std::vector<tokenVector> &args,
                     const int arg_);

      virtual bool expand(tokenVector &newTokens,
                          token_t *source,
                          std::vector<tokenVector> &args);
    };

    class macroStringify : public macroToken {
    public:
      macroToken *token;

      macroStringify(macroToken *token_);
      ~macroStringify();

      virtual bool expand(tokenVector &newTokens,
                          token_t *source,
                          std::vector<tokenVector> &args);
    };

    class macroConcat : public macroToken {
    public:
      macroTokenVector_t tokens;

      macroConcat(const macroTokenVector_t &tokens_);
      ~macroConcat();

      virtual bool expand(tokenVector &newTokens,
                          token_t *source,
                          std::vector<tokenVector> &args);
    };
    //==================================

    //---[ Macro ]----------------------
    class macro_t {
    public:
      static const std::string VA_ARGS;

      preprocessor &pp;
      identifierToken &thisToken;

      bool hasVarArgs;
      intTrie argNames;
      macroTokenVector_t macroTokens;

      macro_t(preprocessor &pp_,
              identifierToken &thisToken_);

      macro_t(preprocessor &pp_,
              const std::string &name_);

      virtual ~macro_t();

      inline int argCount() const {
        return (argNames.size() - hasVarArgs);
      }

      inline bool isFunctionLike() const {
        return ((argCount() > 0) || hasVarArgs);
      }

      inline const std::string& name() const {
        return thisToken.value;
      }

      void loadDefinition();

      void loadFunctionLikeDefinition(tokenVector &tokens);
      bool loadDefinitionArgument(token_t *token);

      void setDefinition(tokenVector &tokens);
      void setMacroTokens(tokenVector &tokens);

      const operator_t* getOperator(macroToken *mToken);
      bool isHash(macroToken *mToken);
      bool isHashhash(macroToken *mToken);

      void stringifyMacroTokens();
      void concatMacroTokens();

      virtual void expand(tokenVector &tokens,
                          identifierToken &source);

      bool loadArgs(identifierToken &source,
                    std::vector<tokenVector> &args);
      bool checkArgs(identifierToken &source,
                     std::vector<tokenVector> &args);

      void printError(token_t *token,
                      const std::string &message);
      void printError(macroToken *mToken,
                      const std::string &message);

      static macro_t* defineBuiltin(preprocessor &pp_,
                                    const std::string &name_,
                                    const std::string &contents);

      static macro_t* define(preprocessor &pp_,
                             fileOrigin origin,
                             const std::string &name_,
                             const std::string &contents);
    };
  }
}

#endif
