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
#include <cstring>

#include "occa/defines.hpp"
#include "occa/types.hpp"

#include "occa/tools/string.hpp"
#include "macro.hpp"
#include "tokenizer.hpp"
#include "preprocessor.hpp"

namespace occa {
  namespace lang {
    const std::string macro_t::VA_ARGS = "__VA_ARGS__";

    //---[ Macro Tokens ]---------------
    macroToken::~macroToken() {}

    std::string macroToken::stringifyTokens(tokenVector &tokens,
                                            const bool addSpaces) {
      std::stringstream ss;
      const int tokenCount = (int) tokens.size();
      for (int i = 0; i < tokenCount; ++i) {
        tokens[i]->print(ss);
        if (addSpaces) {
          ss << ' ';
        }
      }
      return ss.str();
    }

    macroValue::macroValue(token_t *token_) :
      token(token_) {}

    macroValue::~macroValue() {
      delete token;
    }

    void macroValue::expandTokens(tokenVector &newTokens,
                                  token_t *source,
                                  std::vector<tokenVector> &args) {
      newTokens.push_back(token->clone());
    }

    macroArgument::macroArgument(const int arg_) :
      arg(arg_) {}

    void macroArgument::expandTokens(tokenVector &newTokens,
                                     token_t *source,
                                     std::vector<tokenVector> &args) {
      tokenVector &argTokens = args[arg];
      const int tokenCount = (int) argTokens.size();
      for (int i = 0; i < tokenCount; ++i) {
        newTokens.push_back(argTokens[i]);
      }
    }

    macroStringify::macroStringify(macroToken *token_) :
      token(token_) {}

    macroStringify::~macroStringify() {
      delete token;
    }

    void macroStringify::expandTokens(tokenVector &newTokens,
                                      token_t *source,
                                      std::vector<tokenVector> &args) {
      // Get tokens to stringify
      token->expandTokens(newTokens, source, args);
      const std::string rawValue = stringifyTokens(newTokens, true);

      // Escape double quotes
      std::string value = "\"";
      value += escape(rawValue, '"');
      value += '"';

      // Create token
      newTokens.clear();
      tokenizer::tokenize(newTokens,
                          source->origin,
                          value);
    }

    macroConcat::macroConcat(const std::vector<macroToken*> &tokens_) :
      tokens(tokens_) {}

    macroConcat::~macroConcat() {
      const int macroTokenCount = (int) tokens.size();
      for (int i = 0; i < macroTokenCount; ++i) {
        delete tokens[i];
      }
      tokens.clear();
    }

    void macroConcat::expandTokens(tokenVector &newTokens,
                                   token_t *source,
                                   std::vector<tokenVector> &args) {
      // Evaluate all parts
      const int macroTokenCount = (int) tokens.size();
      for (int i = 0; i < macroTokenCount; ++i) {
        tokens[i]->expandTokens(newTokens, source, args);
      }

      // Combine tokens to create one token identifier
      const std::string newToken = stringifyTokens(newTokens, false);

      // Create token
      newTokens.clear();
      tokenizer::tokenize(newTokens,
                          source->origin,
                          newToken);
      newTokens.clear();
    }
    //==================================

    //---[ Macro ]----------------------
    macro_t::macro_t(preprocessor &pp_,
                     identifierToken &thisToken_) :
      pp(pp_),
      thisToken(thisToken_
                .clone()
                ->to<identifierToken>()),
      argCount(-1),
      hasVarArgs(false) {}


    macro_t::macro_t(preprocessor &pp_,
                     const std::string &name_) :
      pp(pp_),
      thisToken(*(new identifierToken(source::builtin, name_))),
      argCount(-1),
      hasVarArgs(false) {}

    macro_t::macro_t(const macro_t &other) :
      pp(other.pp),
      thisToken(other.thisToken),
      argCount(other.argCount),
      hasVarArgs(other.hasVarArgs),
      macroTokens(other.macroTokens) {}

    macro_t::~macro_t() {
      delete &thisToken;
      const int tokens = (int) macroTokens.size();
      for (int i = 0; i < tokens; ++i) {
        delete macroTokens[i];
      }
      macroTokens.clear();
    }

    bool macro_t::expandTokens(identifierToken &source) {
      // Assumes ( has been checked
      std::vector<tokenVector> args;
      bool succeeded = loadArgs(source, args);
      if (!succeeded) {
        return false;
      }
      // Expand tokens
      tokenVector newTokens;
      const int macroTokenCount = (int) macroTokens.size();
      for (int i = 0; i < macroTokenCount; ++i) {
        macroTokens[i]->expandTokens(newTokens,
                                     &source,
                                     args);
      }
      // Push new tokens
      const int newTokenCount = (int) newTokens.size();
      for (int i = 0; i < newTokenCount; ++i) {
        pp.push(newTokens[i]);
      }
      return true;
    }

    bool macro_t::loadArgs(identifierToken &source,
                           std::vector<tokenVector> &args) {
      if (!isFunctionLike()) {
        return true;
      }

      // Clear args
      args.clear();
      for (int i = 0; i < argCount; ++i) {
        args.push_back(tokenVector());
      }

      int argIndex = 0;
      token_t *token = NULL;
      while (true) {
        pp >> token;
        if (!token) {
          printError(&source,
                     "Not able to find closing )");
          break;
        }
        if (token->type() != tokenType::op) {
          // Add token to current arg
          args[argIndex].push_back(token);
          continue;
        }

        // Check for closing )
        opType_t opType = token->to<operatorToken>().op.opType;
        if (opType == operatorType::parenthesesEnd) {
          return true;
        }

        // Check for comma
        if (opType != operatorType::comma) {
          // Add token to current arg
          args[argIndex].push_back(token);
        }

        // Load next argument and check
        ++argIndex;
        args.push_back(tokenVector());
        if (!hasVarArgs && (argIndex >= argCount)) {
          if (argCount) {
            std::stringstream ss;
            ss << "Too many arguments, expected "
               << argCount << " argument(s)";
            printError(token, ss.str());
          } else {
            printError(token,
                       "Macro does not take arguments");
          }
          break;
        }
      }
      return false;
    }

    void macro_t::printError(token_t *token,
                             const std::string &message) {
      fileOrigin fp = thisToken.origin;
      fp.push(false,
              *(token->origin.file),
              token->origin.position);
      fp.printError(message);
    }

#if 0
    macro_t macro_t::define(const std::string &name_,
                            const std::string &contents) {
      fileOrigin origin(source::builtin,
                        contents.c_str());
      return define(origin, name_, contents);
    }

    macro_t macro_t::define(fileOrigin origin,
                            const std::string &name_,
                            const std::string &contents) {
      tokenVector tokens;
      tokenizer::tokenize(tokens,
                          origin,
                          contents);
    }
#endif
    //==================================
  }
}
