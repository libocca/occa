#include <occa/internal/lang/loaders/attributeLoader.hpp>
#include <occa/internal/lang/parser.hpp>
#include <occa/internal/lang/statement/statement.hpp>
#include <occa/internal/lang/statementContext.hpp>
#include <occa/internal/lang/statementPeeker.hpp>
#include <occa/internal/lang/tokenContext.hpp>

namespace occa {
  namespace lang {
    statementPeeker_t::statementPeeker_t(tokenContext_t &tokenContext_,
                                         statementContext_t &smntContext_,
                                         parser_t &parser_,
                                         nameToAttributeMap &attributeMap_) :
      tokenContext(tokenContext_),
      smntContext(smntContext_),
      parser(parser_),
      attributeMap(attributeMap_),
      success(true),
      lastPeek(0),
      lastPeekPosition(-1) {

      // Setup simple keyword -> statement peeks
      keywordPeek[keywordType::qualifier]   = statementType::declaration;
      keywordPeek[keywordType::type]        = statementType::declaration;
      keywordPeek[keywordType::variable]    = statementType::expression;
      keywordPeek[keywordType::function]    = statementType::expression;
      keywordPeek[keywordType::if_]         = statementType::if_;
      keywordPeek[keywordType::switch_]     = statementType::switch_;
      keywordPeek[keywordType::case_]       = statementType::case_;
      keywordPeek[keywordType::default_]    = statementType::default_;
      keywordPeek[keywordType::for_]        = statementType::for_;
      keywordPeek[keywordType::while_]      = statementType::while_;
      keywordPeek[keywordType::do_]         = statementType::while_;
      keywordPeek[keywordType::break_]      = statementType::break_;
      keywordPeek[keywordType::continue_]   = statementType::continue_;
      keywordPeek[keywordType::return_]     = statementType::return_;
      keywordPeek[keywordType::public_]     = statementType::classAccess;
      keywordPeek[keywordType::protected_]  = statementType::classAccess;
      keywordPeek[keywordType::private_]    = statementType::classAccess;
      keywordPeek[keywordType::namespace_]  = statementType::namespace_;
      keywordPeek[keywordType::goto_]       = statementType::goto_;
    }

    void statementPeeker_t::clear() {
      lastPeek = 0;
      lastPeekPosition = -1;
    }

    bool statementPeeker_t::peek(attributeTokenMap &attributes,
                                 int &statementType) {
      const int tokenContextPosition = tokenContext.position();
      if (lastPeekPosition != tokenContextPosition) {
        success = true;
        setupPeek(attributes);
        lastPeek = (success
                    ? uncachedPeek()
                    : statementType::none);
        lastPeekPosition = tokenContextPosition;
      }
      statementType = lastPeek;
      return success;
    }

    int statementPeeker_t::uncachedPeek() {
      const int tokens = tokenContext.size();
      if (!tokens) {
        return statementType::none;
      }

      int tokenIndex = 0;

      while (success && (tokenIndex < tokens)) {
        token_t *token = tokenContext[tokenIndex];
        const int tokenType = token->type();

        if (tokenType & tokenType::identifier) {
          return peekIdentifier(tokenIndex);
        }

        if (tokenType & tokenType::op) {
          return peekOperator(tokenIndex);
        }

        if (tokenType & (tokenType::primitive |
                         tokenType::string    |
                         tokenType::char_)) {
          return statementType::expression;
        }

        if (tokenType & tokenType::directive) {
          return statementType::directive;
        }

        if (tokenType & tokenType::pragma) {
          return statementType::pragma;
        }

        ++tokenIndex;
      }

      return statementType::none;
    }

    void statementPeeker_t::setupPeek(attributeTokenMap &attributes) {
      int tokenContextPos = -1;
      while (success &&
             tokenContext.size() &&
             (tokenContextPos != tokenContext.position())) {
        tokenContextPos = tokenContext.position();
        success &= loadAttributes(tokenContext,
                                  smntContext,
                                  parser,
                                  attributeMap,
                                  attributes);
      }
    }

    int statementPeeker_t::peekIdentifier(const int tokenIndex) {
      token_t *token = tokenContext[tokenIndex];
      int kType = parser.keywords.get(smntContext, token).type();

      if (kType & keywordType::none) {
        // Test for : for it to be a goto label
        if (isGotoLabel(tokenIndex + 1)) {
          return statementType::gotoLabel;
        }
        // TODO: Make sure it's a defined variable
        return statementType::expression;
      }

      const int sType = keywordPeek[kType];
      if (sType) {
        return sType;
      }

      if (kType & keywordType::else_) {
        keyword_t &nextKeyword = parser.keywords.get(smntContext,
                                                     tokenContext[tokenIndex + 1]);
        if ((nextKeyword.type() & keywordType::if_)) {
          return statementType::elif_;
        }
        return statementType::else_;
      }

      token->printError("Unknown identifier");
      success = false;
      return statementType::none;
    }

    bool statementPeeker_t::isGotoLabel(const int tokenIndex) {
      return (
        token_t::safeOperatorType(tokenContext[tokenIndex]) & operatorType::colon
      );
    }

    int statementPeeker_t::peekOperator(const int tokenIndex) {
      const opType_t opType = token_t::safeOperatorType(tokenContext[tokenIndex]);
      if (opType & operatorType::braceStart) {
        return statementType::block;
      }
      if (opType & operatorType::semicolon) {
        return statementType::empty;
      }
      return statementType::expression;
    }
  }
}
