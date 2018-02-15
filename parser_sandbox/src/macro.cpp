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

    macroRawToken::macroRawToken(token_t *token_) :
      token(token_) {}

    macroRawToken::~macroRawToken() {
      delete token;
    }

    bool macroRawToken::expand(tokenVector &newTokens,
                               token_t *source,
                               std::vector<tokenVector> &args) {
      newTokens.push_back(token->clone());
      return true;
    }

    macroArgument::macroArgument(const int arg_) :
      arg(arg_) {}

    bool macroArgument::expand(tokenVector &newTokens,
                               token_t *source,
                               std::vector<tokenVector> &args) {
      tokenVector &argTokens = args[arg];
      const int tokenCount = (int) argTokens.size();
      for (int i = 0; i < tokenCount; ++i) {
        newTokens.push_back(argTokens[i]);
      }
      return true;
    }

    macroStringify::macroStringify(macroToken *token_) :
      token(token_) {}

    macroStringify::~macroStringify() {
      delete token;
    }

    bool macroStringify::expand(tokenVector &newTokens,
                                token_t *source,
                                std::vector<tokenVector> &args) {
      // Get tokens to stringify
      bool success = token->expand(newTokens, source, args);
      if (!success) {
        return false;
      }

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
      return (newTokens.size() == 1);
    }

    macroConcat::macroConcat(const macroTokenVector_t &tokens_) :
      tokens(tokens_) {}

    macroConcat::~macroConcat() {
      const int macroTokenCount = (int) tokens.size();
      for (int i = 0; i < macroTokenCount; ++i) {
        delete tokens[i];
      }
      tokens.clear();
    }

    bool macroConcat::expand(tokenVector &newTokens,
                             token_t *source,
                             std::vector<tokenVector> &args) {
      // Evaluate all parts
      const int macroTokenCount = (int) tokens.size();
      for (int i = 0; i < macroTokenCount; ++i) {
        bool success = tokens[i]->expand(newTokens, source, args);
        if (!success) {
          return false;
        }
      }

      // Combine tokens to create one token identifier
      const std::string newToken = stringifyTokens(newTokens, false);

      // Create token
      newTokens.clear();
      tokenizer::tokenize(newTokens,
                          source->origin,
                          newToken);
      return (newTokens.size() == 1);
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
      thisToken(*(new identifierToken(originSource::builtin, name_))),
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
      clearMacroTokens();
    }

    void macro_t::clearMacroTokens() {
      const int tokens = (int) macroTokens.size();
      for (int i = 0; i < tokens; ++i) {
        delete macroTokens[i];
      }
      macroTokens.clear();
    }

    void macro_t::setDefinition(tokenVector &tokens) {
      clearMacroTokens();
      setMacroTokens(tokens);
      stringifyMacroTokens();
      concatMacroTokens();
    }

    void macro_t::setMacroTokens(tokenVector &tokens) {
      const int tokenCount = (int) tokens.size();
      if (!tokenCount) {
        return;
      }

      for (int i = 0; i < tokenCount; ++i) {
        token_t *token = tokens[i];
        if (!token) {
          continue;
        }
        if (token->type() & tokenType::identifier) {
          const std::string &value = token->to<identifierToken>().value;
          intTrie::result_t result = argNames.get(value);
          if (result.success()) {
            macroTokens.push_back(
              new macroArgument(result.value())
            );
            continue;
          }
        }
        macroTokens.push_back(new macroRawToken(token));
      }
    }

    void macro_t::stringifyMacroTokens() {
      const int tokenCount = (int) macroTokens.size();
      if (!tokenCount) {
        return;
      }

      macroTokenVector_t newMacroTokens;
      for (int i = 0; i < tokenCount; ++i) {
        macroRawToken *mToken = dynamic_cast<macroRawToken*>(macroTokens[i]);
        if (!mToken) {
          newMacroTokens.push_back(mToken);
          continue;
        }
        token_t *token = mToken->token;
        if (!(token->type() & tokenType::op)) {
          newMacroTokens.push_back(mToken);
          continue;
        }
        operatorToken &opToken = token->to<operatorToken>();
        if (!(opToken.op.opType & operatorType::hash)) {
          newMacroTokens.push_back(mToken);
          continue;
        }
        delete mToken;
        if (i < (tokenCount - 1)) {
          newMacroTokens.push_back(new macroStringify(mToken));
        }
      }

      macroTokens = newMacroTokens;
    }

    void macro_t::concatMacroTokens() {
      const int tokenCount = (int) macroTokens.size();
      if (!tokenCount) {
        return;
      }

      macroTokenVector_t newMacroTokens;
      macroTokenVector_t concatMacroTokens;
      macroToken *prevToken = NULL;
      for (int i = 0; i < tokenCount; ++i) {
        macroRawToken *mToken = dynamic_cast<macroRawToken*>(macroTokens[i]);

        token_t *token = (mToken
                          ? mToken->token
                          : NULL);

        operatorToken *opToken = ((token_t::safeType(token) & tokenType::op)
                                  ? &(token->to<operatorToken>())
                                  : NULL);

        // First test the case without ##
        if (!opToken ||
            !(opToken->op.opType & operatorType::hashhash)) {
          prevToken = mToken;
          continue;
        }
        // Test if ## occurred in the beginning or ending
        if (i == 0) {
          token->printError("Macro definition cannot start with ##");
          macroTokens.clear();
          return;
        }
        else if (i == (tokenCount - 1)) {
          token->printError("Macro definition cannot end with ##");
          macroTokens.clear();
          return;
        }
        // Add concat tokens
        concatMacroTokens.push_back(prevToken);
        delete prevToken;
        delete mToken;
        prevToken = NULL;
      }
      // Finish the last token if needed and
      //   update the macro tokens
      if (prevToken) {
        if (concatMacroTokens.size()) {
          newMacroTokens.push_back(new macroConcat(concatMacroTokens));
          concatMacroTokens.clear();
        }
        newMacroTokens.push_back(prevToken);
      }
      macroTokens = newMacroTokens;
    }

    // Assumes ( has already been loaded and verified
    bool macro_t::expand(identifierToken &source,
                         tokenVector &expandedTokens) {
      std::vector<tokenVector> args;
      bool succeeded = loadArgs(source, args);
      if (!succeeded) {
        return false;
      }

      // Expand tokens
      const int macroTokenCount = (int) macroTokens.size();
      for (int i = 0; i < macroTokenCount; ++i) {
        succeeded = macroTokens[i]->expand(expandedTokens,
                                           &source,
                                           args);
        if (!succeeded) {
          return false;
        }
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

    macro_t* macro_t::defineBuiltin(preprocessor &pp_,
                                    const std::string &name_,
                                    const std::string &contents) {
      fileOrigin origin(originSource::builtin,
                        contents.c_str());
      return define(pp_, origin, name_, contents);
    }

    macro_t* macro_t::define(preprocessor &pp_,
                             fileOrigin origin,
                             const std::string &name_,
                             const std::string &contents) {
      std::string source = name_;
      source += ' ';
      source += contents;

      tokenVector tokens;
      tokenizer::tokenize(tokens,
                          origin,
                          source);

      const int tokenCount = (int) tokens.size();
      if (tokenCount == 0) {
        origin.printError("Expected an identifier");
        return NULL;
      }

      identifierToken &macroToken = tokens[0]->to<identifierToken>();
      tokens.erase(tokens.begin());

      macro_t &macro = *(new macro_t(pp_, macroToken));
      macro.setDefinition(tokens);

      return &macro;
    }
    //==================================
  }
}
