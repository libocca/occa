#include <cstring>

#include <occa/defines.hpp>
#include <occa/types.hpp>

#include <occa/internal/utils/string.hpp>
#include <occa/internal/lang/macro.hpp>
#include <occa/internal/lang/tokenizer.hpp>
#include <occa/internal/lang/preprocessor.hpp>

namespace occa {
  namespace lang {
    const std::string macro_t::VA_ARGS = "__VA_ARGS__";

    //---[ Helper Methods ]-------------
    void freeTokenVectors(std::vector<tokenVector> &tokenVectors) {
      const int vectorCount = (int) tokenVectors.size();
      for (int i = 0; i < vectorCount; ++i) {
        freeTokenVector(tokenVectors[i]);
      }
      tokenVectors.clear();
    }

    void freeTokenVector(macroTokenVector &mTokens) {
      const int macroTokenCount = (int) mTokens.size();
      for (int i = 0; i < macroTokenCount; ++i) {
        delete mTokens[i];
      }
      mTokens.clear();
    }

    void cloneMacroTokenVector(macroTokenVector &newTokens,
                               const macroTokenVector &tokens) {
      const int count = (int) tokens.size();
      newTokens.reserve(count);

      for (int i = 0; i < count; ++i) {
        newTokens.push_back(tokens[i]->clone());
      }
    }
    //==================================

    //---[ Macro Tokens ]---------------
    macroToken::macroToken(preprocessor_t &pp_,
                           token_t *thisToken_) :
      pp(pp_),
      thisToken(thisToken_->clone()) {}

    macroToken::~macroToken(){
      delete thisToken;
    }

    token_t* macroToken::cloneToken(token_t *token,
                                    token_t *sourceToken) {
      if (!token) {
        return NULL;
      }
      token_t *newToken = token->clone();
      newToken->origin.push(false, sourceToken->origin);
      return newToken;
    }

    void macroToken::expandDefinedTokens(tokenVector &tokens) {
      tokenVector expandedTokens;
      pp.expandDefinedTokens(tokens, expandedTokens);

      // Update tokens with expanded tokens
      tokens.swap(expandedTokens);

      // Free old tokens
      freeTokenVector(expandedTokens);
    }

    macroRawToken::macroRawToken(preprocessor_t &pp_,
                                 token_t *thisToken_) :
      macroToken(pp_, thisToken_) {}

    macroToken* macroRawToken::clone() const {
      return new macroRawToken(pp, thisToken);
    }

    bool macroRawToken::expand(tokenVector &newTokens,
                               token_t *source,
                               std::vector<tokenVector> &args) {
      newTokens.push_back(
        cloneToken(thisToken, source)
      );
      return true;
    }

    macroArgument::macroArgument(preprocessor_t &pp_,
                                 token_t *thisToken_,
                                 const int arg_,
                                 const int argc_) :
      macroToken(pp_, thisToken_),
      arg(arg_),
      argc(argc_) {}

    macroArgument::~macroArgument() {}

    macroToken* macroArgument::clone() const {
      return new macroArgument(pp, thisToken, arg, argc);
    }

    void macroArgument::expandArg(tokenVector &newTokens,
                                  token_t *source,
                                  std::vector<tokenVector> &args,
                                  const int arg_) {
      tokenVector &argTokens = args[arg_];
      const int tokenCount = (int) argTokens.size();
      for (int i = 0; i < tokenCount; ++i) {
        newTokens.push_back(
          cloneToken(argTokens[i], source)
        );
      }
    }

    bool macroArgument::expand(tokenVector &newTokens,
                               token_t *source,
                               std::vector<tokenVector> &args) {
      if (arg >= 0) {
        expandArg(newTokens, source, args, arg);
      } else {
        // __VA_ARGS__
        const int realArgc = (int) args.size();
        for (int i = argc; i < realArgc; ++i) {
          expandArg(newTokens, source, args, i);
        }
      }

      return true;
    }

    macroStringify::macroStringify(preprocessor_t &pp_,
                                   macroToken *token_) :
      macroToken(pp_, token_->thisToken),
      token(token_) {}

    macroStringify::~macroStringify() {
      delete token;
    }

    macroToken* macroStringify::clone() const {
      return new macroStringify(pp, token);
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

      // Make sure to expand defined variables
      expandDefinedTokens(stringTokens);

      const std::string rawValue = stringifyTokens(stringTokens, true);

      // Escape double quotes
      std::string value = "\"";
      value += escape(rawValue, '"');
      value += '"';

      // Create token
      freeTokenVector(stringTokens);
      tokenizer_t::tokenize(stringTokens,
                            source->origin,
                            value);

      if (stringTokens.size() != 1) {
        source->origin
          .from(false, thisToken->origin)
          .printError("Unable to stringify token");
        ++pp.errors;

        freeTokenVector(stringTokens);
        return false;
      }

      newTokens.push_back(stringTokens[0]);
      return true;
    }

    macroConcat::macroConcat(preprocessor_t &pp_,
                             const macroTokenVector &tokens_) :
      macroToken(pp_, tokens_[0]->thisToken),
      tokens(tokens_) {}

    macroConcat::~macroConcat() {
      freeTokenVector(tokens);
    }

    macroToken* macroConcat::clone() const {
      macroTokenVector newTokens;
      cloneMacroTokenVector(newTokens, tokens);
      return new macroConcat(pp, newTokens);
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

      // Make sure to expand defined variables
      expandDefinedTokens(concatTokens);

      // Combine tokens to create one token identifier
      const std::string concatValue = stringifyTokens(concatTokens, false);

      // Create token
      freeTokenVector(concatTokens);
      tokenizer_t::tokenize(concatTokens,
                            source->origin,
                            concatValue);

      if (concatTokens.size() != 1) {
        concatTokens[0]->origin
          .from(false, thisToken->origin)
          .printError("Unable to concat tokens");
        ++pp.errors;

        freeTokenVector(concatTokens);
        return false;
      }

      newTokens.push_back(concatTokens[0]);
      return true;
    }
    //==================================

    //---[ Macro ]----------------------
    macro_t::macro_t(preprocessor_t &pp_,
                     identifierToken &thisToken_,
                     const bool isBuiltin_,
                     const bool isFunctionLike_,
                     const bool hasVarArgs_) :
      pp(pp_),
      thisToken(thisToken_
                .clone()
                ->to<identifierToken>()),
      isBuiltin(isBuiltin_),
      isFunctionLike(isFunctionLike_),
      hasVarArgs(hasVarArgs_) {
      setupTokenOrigin();
    }


    macro_t::macro_t(preprocessor_t &pp_,
                     const std::string &name_) :
      pp(pp_),
      thisToken(*(new identifierToken(originSource::builtin, name_))),
      isBuiltin(true),
      isFunctionLike(false),
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

    macro_t& macro_t::clone(preprocessor_t &pp_) const {
      macro_t &macro = *(new macro_t(pp_,
                                     thisToken,
                                     isBuiltin,
                                     isFunctionLike,
                                     hasVarArgs));
      cloneMacroTokenVector(macro.macroTokens,
                            macroTokens);
      return macro;
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
      if (!(opToken.opType() & operatorType::parenthesesStart)) {
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

      isFunctionLike = true;
      loadFunctionLikeDefinition(tokens);
      freeTokenVector(tokens);
    }

    void macro_t::loadFunctionLikeDefinition(tokenVector &tokens) {
      const int tokenCount = (int) tokens.size();
      int index = 0;
      bool loadedArgs = false;

      // No arguments
      if (tokenCount &&
          (tokens[0]->getOpType() & operatorType::parenthesesEnd)) {
        delete tokens[0];
        tokens.erase(tokens.begin());
        setDefinition(tokens);
        return;
      }

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
          opType_t opType = token->getOpType();
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
          errorOn(token,
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
        errorOn(token,
                "Cannot have arguments after ...");
        return false;
      }

      bool isArg = (token->type() & tokenType::identifier);
      if (!isArg &&
          (token->type() & tokenType::op)) {
        opType_t opType = token->to<operatorToken>().opType();
        if (opType & operatorType::ellipsis) {
          isArg = true;
          hasVarArgs = true;
        }
      }

      if (!isArg) {
        errorOn(token,
                "Expected an identifier as a macro argument");
        return false;
      }

      if (!hasVarArgs) {
        const int arg = (int) argNames.size();
        argNames[token->to<identifierToken>().value] = arg;
      } else {
        argNames[VA_ARGS] = -1;
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
          intMapIterator it = argNames.find(value);
          if (it != argNames.end()) {
            macroTokens.push_back(
              new macroArgument(pp,
                                token,
                                it->second,
                                argc)
            );
            continue;
          }
        }
        macroTokens.push_back(new macroRawToken(pp, token));
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

      return token->to<operatorToken>().op;
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

      macroTokenVector newMacroTokens;
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
          newMacroTokens.push_back(new macroStringify(pp, argToken));
          continue;
        }

        // We're going to concat # instead of using
        //   it for stringification
        if (isHashhash(macroTokens[i])) {
          newMacroTokens.push_back(macroTokens[i - 1]);
          newMacroTokens.push_back(macroTokens[i]);
          continue;
        }

        errorOn(macroTokens[i],
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
        errorOn(macroTokens[0],
                "Macro definition cannot start with ##");
        freeTokenVector(macroTokens);
        return;
      }
      if ((tokenCount > 1) &&
          isHashhash(macroTokens[tokenCount - 1])) {
        errorOn(macroTokens[tokenCount - 1],
                "Macro definition cannot end with ##");
        freeTokenVector(macroTokens);
        return;
      }

      macroTokenVector newMacroTokens;
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
        macroTokenVector concatTokens;
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
        newMacroTokens.push_back(new macroConcat(pp, concatTokens));
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
      if (!isFunctionLike) {
        return true;
      }

      tokenVector argTokens;

      // Count the initial [(] token
      int parenthesesCount = 1;

      // Pull all of the argument tokens first
      while (true) {
        token_t *token = NULL;
        pp >> token;

        if (!token) {
          errorOn(&source,
                  "Not able to find a closing )");
          break;
        }

        // Check for closing ) first
        if (token->type() & tokenType::op) {
          opType_t opType = token->to<operatorToken>().opType();
          if (opType & (operatorType::parenthesesStart |
                        operatorType::parenthesesEnd)) {
            if (opType & operatorType::parenthesesStart) {
              ++parenthesesCount;
            } else {
              --parenthesesCount;
              if (!parenthesesCount) {
                delete token;
                break;
              }
            }
          }
        }

        // Push tokens between the [(] and [)] tokens
        argTokens.push_back(token);
      }

      const int argc = argCount();
      int argIndex = 0;
      parenthesesCount = 0;

      const int tokenCount = (int) argTokens.size();
      for (int i = 0; i < tokenCount; ++i) {
        token_t *token = argTokens[i];
        // Make it easy to free the vector if something goes wrong
        argTokens[i] = NULL;

        const opType_t opType = token_t::safeOperatorType(token);

        // Check if we're adding a new argument and that we don't go out of bounds
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
              errorOn(token, ss.str());
            } else {
              errorOn(token,
                      "Macro does not take arguments");
            }
            freeTokenVector(argTokens);
            return false;
          }
        }

        if ((opType == operatorType::comma) && !parenthesesCount) {
          // Starting next argument
          ++argIndex;
          delete token;
          continue;
        }

        if (opType == operatorType::parenthesesStart) {
          ++parenthesesCount;
        } else if (opType == operatorType::parenthesesEnd) {
          --parenthesesCount;
        }

        args[argIndex].push_back(token);
      }

      return true;
    }

    bool macro_t::checkArgs(identifierToken &source,
                            std::vector<tokenVector> &args) {
      const int realArgc = argCount();
      const int argc     = (int) args.size();

      if ((argc < realArgc) ||
          ((argc > realArgc) && !hasVarArgs)) {
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

        errorOn(&source, ss.str());
        return false;
      }

      return true;
    }

    void macro_t::errorOn(token_t *token,
                          const std::string &message) {
      token->origin
        .from(false, thisToken.origin)
        .printError(message);
      ++pp.errors;
    }

    void macro_t::errorOn(macroToken *mToken,
                          const std::string &message) {
      mToken->thisToken->origin
        .from(false, thisToken.origin)
        .printError(message);
      ++pp.errors;
    }

    macro_t* macro_t::defineBuiltin(preprocessor_t &pp_,
                                    const std::string &name_,
                                    const std::string &contents) {
      fileOrigin origin(originSource::builtin,
                        contents.c_str());
      return define(pp_, origin, name_, contents);
    }

    macro_t* macro_t::define(preprocessor_t &pp_,
                             fileOrigin origin,
                             const std::string &name_,
                             const std::string &contents) {
      std::string source = name_;
      source += ' ';
      source += contents;

      tokenVector tokens;
      tokenizer_t::tokenize(tokens,
                            origin,
                            source);

      const int tokenCount = (int) tokens.size();
      if (tokenCount == 0) {
        origin.printError("Expected an identifier");
        ++pp_.errors;

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
