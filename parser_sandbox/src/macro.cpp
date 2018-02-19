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
    macroToken::macroToken(token_t *thisToken_) :
      thisToken(thisToken_->clone()) {}

    macroToken::~macroToken(){
      delete thisToken;
    }

    macroRawToken::macroRawToken(token_t *thisToken_) :
      macroToken(thisToken_) {}

    bool macroRawToken::expand(tokenVector &newTokens,
                               token_t *source,
                               std::vector<tokenVector> &args) {
      newTokens.push_back(thisToken->clone());
      return true;
    }

    macroArgument::macroArgument(token_t *thisToken_,
                                 const int arg_,
                                 const int argc_) :
      macroToken(thisToken_),
      arg(arg_),
      argc(argc_) {}

    macroArgument::~macroArgument() {}

    void macroArgument::expandArg(tokenVector &newTokens,
                                  std::vector<tokenVector> &args,
                                  const int arg_) {
      tokenVector &argTokens = args[arg_];
      const int tokenCount = (int) argTokens.size();
      for (int i = 0; i < tokenCount; ++i) {
        newTokens.push_back(argTokens[i]->clone());
      }
    }

    bool macroArgument::expand(tokenVector &newTokens,
                               token_t *source,
                               std::vector<tokenVector> &args) {
      if (arg >= 0) {
        expandArg(newTokens, args, arg);
      } else {
        const int realArgc = (int) args.size();
        for (int i = argc; i < realArgc; ++i) {
          expandArg(newTokens, args, i);
        }
      }

      return true;
    }

    macroStringify::macroStringify(macroToken *token_) :
      macroToken(token_->thisToken),
      token(token_) {}

    macroStringify::~macroStringify() {
      delete token;
    }

    bool macroStringify::expand(tokenVector &newTokens,
                                token_t *source,
                                std::vector<tokenVector> &args) {
      // Get tokens to stringify
      tokenVector stringTokens;
      bool success = token->expand(stringTokens, source, args);
      if (!success) {
        freeTokenVector(stringTokens);
        return false;
      }

      const std::string rawValue = stringifyTokens(stringTokens, true);

      // Escape double quotes
      std::string value = "\"";
      value += escape(rawValue, '"');
      value += '"';

      // Create token
      freeTokenVector(stringTokens);
      tokenizer::tokenize(stringTokens,
                          source->origin,
                          value);

      if (stringTokens.size() != 1) {
        source->origin
          .from(false, thisToken->origin)
          .printError("Unable to stringify token");

        freeTokenVector(stringTokens);
        return false;
      }

      newTokens.push_back(stringTokens[0]);
      return true;
    }

    macroConcat::macroConcat(const macroTokenVector_t &tokens_) :
      macroToken(tokens_[0]->thisToken),
      tokens(tokens_) {}

    macroConcat::~macroConcat() {
      freeTokenVector(tokens);
    }

    bool macroConcat::expand(tokenVector &newTokens,
                             token_t *source,
                             std::vector<tokenVector> &args) {
      // Get tokens to concat
      tokenVector concatTokens;
      const int macroTokenCount = (int) tokens.size();
      for (int i = 0; i < macroTokenCount; ++i) {
        bool success = tokens[i]->expand(concatTokens, source, args);
        if (!success) {
          freeTokenVector(concatTokens);
          return false;
        }
      }

      // Combine tokens to create one token identifier
      const std::string concatValue = stringifyTokens(concatTokens, false);

      // Create token
      freeTokenVector(concatTokens);
      tokenizer::tokenize(concatTokens,
                          source->origin,
                          concatValue);

      if (concatTokens.size() != 1) {
        concatTokens[0]->origin
          .from(false, thisToken->origin)
          .printError("Unable to concat tokens");

        freeTokenVector(concatTokens);
        return false;
      }

      newTokens.push_back(concatTokens[0]);
      return true;
    }
    //==================================

    //---[ Helper Methods ]-------------
    void freeTokenVectors(std::vector<tokenVector> &tokenVectors) {
      const int vectorCount = (int) tokenVectors.size();
      for (int i = 0; i < vectorCount; ++i) {
        freeTokenVector(tokenVectors[i]);
      }
      tokenVectors.clear();
    }

    void freeTokenVector(macroTokenVector_t &mTokens) {
      const int macroTokenCount = (int) mTokens.size();
      for (int i = 0; i < macroTokenCount; ++i) {
        delete mTokens[i];
      }
      mTokens.clear();
    }
    //==================================

    //---[ Macro ]----------------------
    macro_t::macro_t(preprocessor &pp_,
                     identifierToken &thisToken_,
                     const bool isBuiltin_) :
      pp(pp_),
      thisToken(thisToken_
                .clone()
                ->to<identifierToken>()),
      isBuiltin(isBuiltin_),
      hasVarArgs(false) {
      setupTokenOrigin();
    }


    macro_t::macro_t(preprocessor &pp_,
                     const std::string &name_) :
      pp(pp_),
      thisToken(*(new identifierToken(originSource::builtin, name_))),
      isBuiltin(true),
      hasVarArgs(false) {
      setupTokenOrigin();
    }

    macro_t::~macro_t() {
      if (isBuiltin) {
        delete [] thisToken.origin.position.start;
      }
      delete &thisToken;
      argNames.clear();
      freeTokenVector(macroTokens);
    }

    void macro_t::setupTokenOrigin() {
      if (!isBuiltin) {
        return;
      }

      const std::string &name_ = name();

      const int chars = (int) name_.size();
      char *c = new char[chars + 1];
      ::memcpy(c, name_.c_str(), chars + 1);

      thisToken.origin.position = filePosition(0, c, c, c + chars);
    }

    macro_t& macro_t::clone(preprocessor &pp_) {
      return *(new macro_t(pp_, thisToken, isBuiltin));
    }

    void macro_t::loadDefinition() {
      tokenVector tokens;
      pp.getLineTokens(tokens);

      // Remove the newline token
      if (tokens.size()) {
        delete tokens.back();
        tokens.pop_back();
      }

      if (!tokens.size()) {
        return;
      }

      token_t *token = tokens[0];
      if (!(token->type() & tokenType::op)) {
        setDefinition(tokens);
        freeTokenVector(tokens);
        return;
      }

      operatorToken &opToken = token->to<operatorToken>();
      if (!(opToken.op.opType & operatorType::parenthesesStart)) {
        setDefinition(tokens);
        freeTokenVector(tokens);
        return;
      }

      // The ( only counts as the start of a function-like
      //   macro if it's directly after the macro name
      dim_t posDistance = thisToken.origin.distanceTo(token->origin);
      if (posDistance != 0) {
        setDefinition(tokens);
        freeTokenVector(tokens);
        return;
      }

      // Remove ( token
      delete token;
      tokens.erase(tokens.begin());

      loadFunctionLikeDefinition(tokens);
      freeTokenVector(tokens);
    }

    void macro_t::loadFunctionLikeDefinition(tokenVector &tokens) {
      const int tokenCount = (int) tokens.size();
      int index = 0;
      bool loadedArgs = false;

      while (index < tokenCount) {
        // Test for arg name
        token_t *token = tokens[index++];
        if (!loadDefinitionArgument(token)) {
          return;
        }

        // Test for ',' or ')'
        token = tokens[index++];
        bool foundOp = (token->type() & tokenType::op);
        if (foundOp) {
          opType_t opType = token->to<operatorToken>().op.opType;
          if (opType & operatorType::comma) {
            continue;
          }
          if (opType & operatorType::parenthesesEnd) {
            loadedArgs = true;
            break;
          }
          foundOp = false;
        }
        if (!foundOp) {
          printError(token,
                     "Expected a , to separate arguments"
                     " or ) to finish the macro definition");
          return;
        }
      }

      if (loadedArgs) {
        for (int i = 0; i < index; ++i) {
          delete tokens[i];
        }
        for (int i = index; i < tokenCount; ++i) {
          tokens[i - index] = tokens[i];
        }
        tokens.resize(tokenCount - index);
        setDefinition(tokens);
      }
    }

    bool macro_t::loadDefinitionArgument(token_t *token) {
      if (hasVarArgs) {
        printError(token,
                   "Cannot have arguments after ...");
        return false;
      }

      bool isArg = (token->type() & tokenType::identifier);
      if (!isArg &&
          (token->type() & tokenType::op)) {
        opType_t opType = token->to<operatorToken>().op.opType;
        if (opType & operatorType::ellipsis) {
          isArg = true;
          hasVarArgs = true;
        }
      }

      if (!isArg) {
        printError(token,
                   "Expected an identifier as a macro argument");
        return false;
      }

      if (!hasVarArgs) {
        argNames.add(token->to<identifierToken>().value,
                     argNames.size());
      } else {
        argNames.add(VA_ARGS,
                     -1);
      }
      return true;
    }

    void macro_t::setDefinition(tokenVector &tokens) {
      setMacroTokens(tokens);
      stringifyMacroTokens();
      concatMacroTokens();
    }

    void macro_t::setMacroTokens(tokenVector &tokens) {
      const int tokenCount = (int) tokens.size();
      if (!tokenCount) {
        return;
      }

      const int argc = argCount();

      for (int i = 0; i < tokenCount; ++i) {
        token_t *token = tokens[i];
        if (!token) {
          continue;
        }
        const int tokenType = token->type();
        if (tokenType & tokenType::identifier) {
          const std::string &value = token->to<identifierToken>().value;
          intTrie::result_t result = argNames.get(value);
          if (result.success()) {
            macroTokens.push_back(
              new macroArgument(token,
                                result.value(),
                                argc)
            );
            continue;
          }
        }
        macroTokens.push_back(new macroRawToken(token));
      }
    }

    const operator_t* macro_t::getOperator(macroToken *mToken) {
      macroRawToken *rawToken = dynamic_cast<macroRawToken*>(mToken);
      if (!rawToken) {
        return NULL;
      }

      token_t *token = rawToken->thisToken;
      if (!(token->type() & tokenType::op)) {
        return NULL;
      }

      return &(token->to<operatorToken>().op);
    }

    bool macro_t::isHash(macroToken *mToken) {
      const operator_t *op = getOperator(mToken);
      return (op &&
              (op->opType & operatorType::hash));
    }

    bool macro_t::isHashhash(macroToken *mToken) {
      const operator_t *op = getOperator(mToken);
      return (op &&
              (op->opType & operatorType::hashhash));
    }

    void macro_t::stringifyMacroTokens() {
      const int tokenCount = (int) macroTokens.size();
      if (!tokenCount) {
        return;
      }

      macroTokenVector_t newMacroTokens;
      for (int i = 0; i < tokenCount; ++i) {
        macroToken *mToken = macroTokens[i];
        if (!isHash(mToken) ||
            (i == (tokenCount - 1))) {
          newMacroTokens.push_back(macroTokens[i]);
          continue;
        }

        ++i;
        macroArgument *argToken = dynamic_cast<macroArgument*>(macroTokens[i]);
        if (argToken) {
          delete mToken;
          newMacroTokens.push_back(new macroStringify(argToken));
          continue;
        }

        // We're going to concat # instead of using
        //   it for stringification
        if (isHashhash(macroTokens[i])) {
          newMacroTokens.push_back(macroTokens[i - 1]);
          newMacroTokens.push_back(macroTokens[i]);
          continue;
        }

        printError(macroTokens[i],
                   "Can only stringify macro arguments");
        for (int j = (i - 1); j < tokenCount; ++j) {
          delete macroTokens[i];
        }
        macroTokens.clear();
        freeTokenVector(newMacroTokens);
        return;
      }

      macroTokens = newMacroTokens;
    }

    void macro_t::concatMacroTokens() {
      const int tokenCount = (int) macroTokens.size();
      if (!tokenCount) {
        return;
      }

      if (isHashhash(macroTokens[0])) {
        printError(macroTokens[0],
                   "Macro definition cannot start with ##");
        freeTokenVector(macroTokens);
        return;
      }
      if ((tokenCount > 1) &&
          isHashhash(macroTokens[tokenCount - 1])) {
        printError(macroTokens[tokenCount - 1],
                   "Macro definition cannot end with ##");
        freeTokenVector(macroTokens);
        return;
      }

      macroTokenVector_t newMacroTokens;
      int lastIndex = 0;
      for (int i = 0; i < tokenCount; ++i) {
        macroToken *mToken = macroTokens[i];
        if (!isHashhash(mToken)) {
          continue;
        }
        // Push tokens between concatenations
        for (int j = lastIndex; j < (i - 1); ++j) {
          newMacroTokens.push_back(macroTokens[j]);
        }

        // Get concat tokens
        macroTokenVector_t concatTokens;
        concatTokens.push_back(macroTokens[i - 1]);
        for (lastIndex = (i + 1); lastIndex < tokenCount; lastIndex += 2) {
          delete macroTokens[lastIndex - 1];

          mToken = macroTokens[lastIndex];
          concatTokens.push_back(mToken);

          if ((lastIndex == (tokenCount - 1)) ||
              !isHashhash(macroTokens[lastIndex + 1])) {
            lastIndex += 1;
            i = (lastIndex - 1);
            break;
          }
        }
        newMacroTokens.push_back(new macroConcat(concatTokens));
      }

      // Push last remaining tokens and copy
      for (int j = lastIndex; j < tokenCount; ++j) {
        newMacroTokens.push_back(macroTokens[j]);
      }
      macroTokens = newMacroTokens;
    }

    // Assumes ( has already been loaded and verified
    void macro_t::expand(tokenVector &tokens,
                         identifierToken &source) {
      std::vector<tokenVector> args;
      if (!loadArgs(source, args) ||
          !checkArgs(source, args)) {
        freeTokenVectors(args);
        return;
      }

      // Expand tokens
      const int macroTokenCount = (int) macroTokens.size();
      for (int i = 0; i < macroTokenCount; ++i) {
        bool succeeded = macroTokens[i]->expand(tokens,
                                                &source,
                                                args);
        if (!succeeded) {
          break;
        }
      }
      freeTokenVectors(args);
    }

    bool macro_t::loadArgs(identifierToken &source,
                           std::vector<tokenVector> &args) {
      if (!isFunctionLike()) {
        return true;
      }

      const int argc = argCount();
      int argIndex = 0;

      token_t *token = NULL;
      while (true) {
        pp >> token;
        if (!token) {
          printError(&source,
                     "Not able to find closing )");
          break;
        }

        // Check for closing ) first
        if (token->type() & tokenType::op) {
          opType_t opType = token->to<operatorToken>().op.opType;
          if (opType == operatorType::parenthesesEnd) {
            delete token;
            return true;
          }
        }

        // Make sure we don't go out of bounds
        if (argIndex >= (int) args.size()) {
          args.push_back(tokenVector());

          if (!hasVarArgs && (argIndex >= argc)) {
            if (argc) {
              std::stringstream ss;
              ss << "Too many arguments, expected "
                 << argc << " argument";
              if (argc > 1) {
                ss << 's';
              }
              printError(token, ss.str());
            } else {
              printError(token,
                         "Macro does not take arguments");
            }
            delete token;
            break;
          }
        }

        if (token->type() != tokenType::op) {
          // Add token to current arg
          args[argIndex].push_back(token);
          continue;
        }

        // Check for comma
        if (token->to<operatorToken>().op.opType != operatorType::comma) {
          // Add token to current arg
          args[argIndex].push_back(token);
          continue;
        }

        // Starting next argument
        ++argIndex;
        delete token;
      }

      return false;
    }

    bool macro_t::checkArgs(identifierToken &source,
                            std::vector<tokenVector> &args) {
      const int realArgc = argCount();
      const int argc     = (int) args.size();

      if (argc < realArgc) {
        std::stringstream ss;
        ss << "Expected " << realArgc << " argument";
        if (realArgc > 1) {
          ss << 's';
        }
        ss << ", instead found ";
        if (argc) {
          ss << argc;
        } else {
          ss << "none";
        }

        printError(&source, ss.str());
        return false;
      }

      return true;
    }

    void macro_t::printError(token_t *token,
                             const std::string &message) {
      token->origin
        .from(false, thisToken.origin)
        .printError(message);
    }

    void macro_t::printError(macroToken *mToken,
                             const std::string &message) {
      mToken->thisToken->origin
        .from(false, thisToken.origin)
        .printError(message);
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
        freeTokenVector(tokens);
        return NULL;
      }

      identifierToken &macroToken = tokens[0]->to<identifierToken>();
      tokens.erase(tokens.begin());

      macro_t &macro = *(new macro_t(pp_, macroToken));
      macro.setDefinition(tokens);
      freeTokenVector(tokens);

      // Macro clones the token
      delete &macroToken;

      return &macro;
    }
    //==================================
  }
}
