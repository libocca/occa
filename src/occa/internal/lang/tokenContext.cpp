#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/keyword.hpp>
#include <occa/internal/lang/loaders/typeLoader.hpp>
#include <occa/internal/lang/statementContext.hpp>
#include <occa/internal/lang/parser.hpp>
#include <occa/internal/lang/token.hpp>
#include <occa/internal/lang/tokenContext.hpp>

namespace occa {
  namespace lang {
    static const int skippableTokenTypes = (
      tokenType::comment
      | tokenType::unknown
      | tokenType::none
    );

    tokenRange::tokenRange() :
      start(0),
      end(0) {}

    tokenRange::tokenRange(const int start_,
                           const int end_) :
      start(start_),
      end(end_) {}

    tokenContext_t::tokenContext_t() {}

    tokenContext_t::~tokenContext_t() {
      clear();
    }

    void tokenContext_t::clear() {
      tp.start = 0;
      tp.end   = 0;

      hasError = false;
      supressErrors = false;

      const int tokenCount = (int) tokens.size();
      for (int i = 0; i < tokenCount; ++i) {
        delete tokens[i];
      }

      tokens.clear();
      tokenIndices.clear();

      pairs.clear();
      semicolons.clear();
      stack.clear();
    }

    // Called once all tokens are set from the parser
    void tokenContext_t::setup(const tokenVector &tokens_) {
      clear();

      tokens = tokens_;
      setupTokenIndices();

      tp.start = 0;
      tp.end   = (int) tokenIndices.size();

      findPairs();
      findSemicolons();
    }

    void tokenContext_t::setupTokenIndices() {
      const int tokenCount = (int) tokens.size();
      for (int i = 0; i < tokenCount; ++i) {
        token_t *token = tokens[i];
        if (!(token->type() & skippableTokenTypes)) {
          tokenIndices.push_back(i);
        }
      }
    }

    void tokenContext_t::findPairs() {
      intVector pairStack;

      const int tokenCount = size();
      for (int i = 0; i < tokenCount; ++i) {
        token_t *token = getToken(i);
        opType_t opType = token->getOpType();
        if (!(opType & operatorType::pair)) {
          continue;
        }
        if (opType & operatorType::pairStart) {
          pairStack.push_back(i);
          continue;
        }

        pairOperator_t &pairEndOp =
          *((pairOperator_t*) token->to<operatorToken>().op);

        // Make sure we have a proper pair
        if (!pairStack.size()) {
          if (!supressErrors) {
            std::stringstream ss;
            ss << "Could not find an opening '"
               << pairEndOp.pairStr
               << '\'';
            token->printError(ss.str());
            hasError = true;
          }
          return;
        }

        // Make sure we close the pair
        const int pairIndex = pairStack.back();
        pairStack.pop_back();
        pairOperator_t &pairStartOp = (
          *((pairOperator_t*) getToken(pairIndex)->to<operatorToken>().op)
        );

        if (pairStartOp.opType != (pairEndOp.opType >> 1)) {
          if (!supressErrors) {
            std::stringstream ss;
            ss << "Could not find a closing '"
               << pairStartOp.pairStr
               << '\'';
            getToken(pairIndex)->printError(ss.str());
            hasError = true;
          }
          return;
        }

        // Store pairs
        pairs[pairIndex] = i;
      }
      // Make sure all pair openers have a close
      if (pairStack.size()) {
        const int pairIndex = pairStack.back();
        pairStack.pop_back();
        pairOperator_t &pairStartOp = (
          *((pairOperator_t*) getToken(pairIndex)->to<operatorToken>().op)
        );

        if (!supressErrors) {
          std::stringstream ss;
          ss << "Could not find a closing '"
             << pairStartOp.pairStr
             << '\'';
          getToken(pairIndex)->printError(ss.str());
          hasError = true;
        }
      }
    }

    void tokenContext_t::findSemicolons() {
      const int tokenCount = size();
      for (int i = 0; i < tokenCount; ++i) {
        token_t *token = getToken(i);
        opType_t opType = token->getOpType();
        if (opType & operatorType::semicolon) {
          semicolons.push_back(i);
        }
      }
    }

    bool tokenContext_t::indexInRange(const int index) const {
      return ((index >= 0) && ((tp.start + index) < tp.end));
    }

    void tokenContext_t::set(const int start) {
      if (indexInRange(start)) {
        tp.start += start;
      } else {
        tp.start = tp.end;
      }
    }

    void tokenContext_t::set(const int start,
                             const int end) {
      if (indexInRange(start)) {
        tp.start += start;
        if (indexInRange(end - start)) {
          tp.end = tp.start + (end - start);
        }
      } else {
        tp.start = tp.end;
      }
    }

    void tokenContext_t::set(const tokenRange &range) {
      set(range.start, range.end);
    }

    void tokenContext_t::push() {
      stack.push_back(tp);
    }

    void tokenContext_t::push(const int start) {
      stack.push_back(tp);
      set(start);
    }

    void tokenContext_t::push(const int start,
                              const int end) {
      stack.push_back(tp);
      set(start, end);
    }

    void tokenContext_t::push(const tokenRange &range) {
      push(range.start, range.end);
    }

    void tokenContext_t::pushPairRange() {
      const int pairEnd = getClosingPair();
      if (pairEnd >= 0) {
        push(1, pairEnd);
      } else {
        OCCA_FORCE_ERROR("Trying to push a pair range without a pair");
      }
    }

    tokenRange tokenContext_t::pop() {
      OCCA_ERROR("Unable to call tokenContext_t::pop",
                 stack.size());

      tokenRange prev = tp;
      tp = stack.back();
      stack.pop_back();

      // Return a relative tokenRange
      const int prevStart = prev.start - tp.start;
      return tokenRange(prevStart,
                        prevStart + (prev.end - prev.start));
    }

    void tokenContext_t::popAndSkip() {
      set(pop().end + 1);
    }

    int tokenContext_t::position() const {
      return tp.start;
    }

    int tokenContext_t::size() const {
      return (tp.end - tp.start);
    }

    void tokenContext_t::getSkippedTokens(tokenVector &skippedTokens,
                                          const int start,
                                          const int end) {
      if (start >= (int) tokenIndices.size()) {
        return;
      }

      const int startNativeIndex = (
        start
        ? tokenIndices[start - 1]
        : 0
      );
      const int endNativeIndex = (
        end < (int) tokenIndices.size()
        ? tokenIndices[end]
        : tp.end
      );

      for (int i = startNativeIndex; i < endNativeIndex; ++i) {
        token_t *token = tokens[i];
        if (token->type() & skippableTokenTypes) {
          skippedTokens.push_back(token);
        }
      }
    }

    token_t* tokenContext_t::getToken(const int index) {
      return tokens[tokenIndices[index]];
    }

    void tokenContext_t::setToken(const int index,
                                  token_t *value) {
      if (!indexInRange(index)) {
        return;
      }
      const int pos = tokenIndices[tp.start + index];
      if (tokens[pos] != value) {
        delete tokens[pos];
        tokens[pos] = value;
      }
    }

    token_t* tokenContext_t::operator [] (const int index) {
      if (!indexInRange(index)) {
        return NULL;
      }
      return getToken(tp.start + index);
    }

    tokenContext_t& tokenContext_t::operator ++ () {
      set(1);
      return *this;
    }

    tokenContext_t& tokenContext_t::operator ++ (int) {
      set(1);
      return *this;
    }

    tokenContext_t& tokenContext_t::operator += (const int offset) {
      set(offset);
      return *this;
    }

    token_t* tokenContext_t::end() {
      if (indexInRange(tp.end - tp.start - 1)) {
        return getToken(tp.end - 1);
      }
      return NULL;
    }

    token_t* tokenContext_t::getPrintToken(const bool atEnd) {
      if (size() == 0) {
        return NULL;
      }

      const int start = tp.start;
      if (atEnd) {
        tp.start = tp.end;
      }

      int offset = 0;
      if (!indexInRange(offset) && (0 < tp.start)) {
        offset = -1;
      }

      token_t *token = getToken(tp.start + offset);
      if (atEnd) {
        // Reset tp.start
        tp.start = start;
      }

      return token;
    }

    void tokenContext_t::printWarning(const std::string &message) {
      token_t *token = getPrintToken(false);
      if (!token) {
        occa::printWarning(io::stderr, "[No Token] " + message);
      } else {
        token->printWarning(message);
      }
    }

    void tokenContext_t::printWarningAtEnd(const std::string &message) {
      token_t *token = getPrintToken(true);
      if (!token) {
        occa::printWarning(io::stderr, "[No Token] " + message);
      } else {
        token->printWarning(message);
      }
    }

    void tokenContext_t::printError(const std::string &message) {
      if (supressErrors) {
        return;
      }
      token_t *token = getPrintToken(false);
      if (!token) {
        occa::printError(io::stderr, "[No Token] " + message);
      } else {
        token->printError(message);
      }
    }

    void tokenContext_t::printErrorAtEnd(const std::string &message) {
      if (supressErrors) {
        return;
      }
      token_t *token = getPrintToken(true);
      if (!token) {
        occa::printError(io::stderr, "[No Token] " + message);
      } else {
        token->printError(message);
      }
    }

    void tokenContext_t::getTokens(tokenVector &tokens_) {
      tokens_.clear();
      tokens_.reserve(tp.end - tp.start);
      for (int i = tp.start; i < tp.end; ++i) {
        tokens_.push_back(getToken(i));
      }
    }

    void tokenContext_t::getAndCloneTokens(tokenVector &tokens_) {
      tokens_.clear();
      tokens_.reserve(tp.end - tp.start);
      for (int i = tp.start; i < tp.end; ++i) {
        tokens_.push_back(getToken(i)->clone());
      }
    }

    int tokenContext_t::getClosingPair() {
      if (!size()) {
        return -1;
      }

      intIntMap::iterator it = pairs.find(tp.start);
      if (it != pairs.end()) {
        return (it->second - tp.start);
      }
      return -1;
    }

    token_t* tokenContext_t::getClosingPairToken() {
      const int endIndex = getClosingPair();
      if (endIndex >= 0) {
        return getToken(tp.start + endIndex);
      }
      return NULL;
    }

    int tokenContext_t::getNextOperator(const opType_t &opType) {
      for (int pos = tp.start; pos < tp.end; ++pos) {
        token_t *token = getToken(pos);
        if (!(token->type() & tokenType::op)) {
          continue;
        }
        const opType_t tokenOpType = (token
                                      ->to<operatorToken>()
                                      .getOpType());

        if (tokenOpType & opType) {
          return (pos - tp.start);
        }
        // Make sure we don't use semicolons inside blocks
        if (tokenOpType & operatorType::pairStart) {
          pos = pairs[pos];
        }
      }
      return -1;
    }

    exprNode* tokenContext_t::parseExpression(statementContext_t &smntContext,
                                              parser_t &parser) {
      return parseExpression(smntContext, parser, 0, size());
    }

    exprNode* tokenContext_t::parseExpression(statementContext_t &smntContext,
                                              parser_t &parser,
                                              const int start,
                                              const int end) {
      push(start, end);
      const int tokenCount = size();
      tokenVector exprTokens;
      exprTokens.reserve(tokenCount);

      // Replace identifier tokens with keywords if they exist
      for (int i = 0; i < tokenCount; ++i) {
        token_t *token = (*this)[i];
        if (token->type() & tokenType::identifier) {
          setToken(i, replaceIdentifier(smntContext,
                                        parser.keywords,
                                        (identifierToken&) *token));
        }
      }

      while (size()) {
        token_t *token = (*this)[0];
        int tokenType = token->type();

        // Ignore comments
        if (tokenType & tokenType::comment) {
          set(1);
          continue;
        }

        if (!(tokenType & (tokenType::qualifier |
                           tokenType::type))) {
          set(1);
          exprTokens.push_back(token->clone());
          continue;
        }

        vartype_t vartype;
        if (!loadType(*this, smntContext, parser, vartype)) {
          pop();
          freeTokenVector(exprTokens);
          return NULL;
        }

        exprTokens.push_back(new vartypeToken(token->origin,
                                          vartype));
      }
      pop();

      return expressionParser::parse(exprTokens);
    }

    token_t* tokenContext_t::replaceIdentifier(statementContext_t &smntContext,
                                               const keywords_t &keywords,
                                               identifierToken &identifier) {
      keyword_t &keyword = keywords.get(smntContext, &identifier);
      const int kType = keyword.type();

      if (!(kType & (keywordType::qualifier |
                     keywordType::type      |
                     keywordType::variable  |
                     keywordType::function))) {
        return &identifier;
      }

      if (kType & keywordType::qualifier) {
        return new qualifierToken(identifier.origin,
                                  ((qualifierKeyword&) keyword).qualifier);
      }
      if (kType & keywordType::variable) {
        return new variableToken(identifier.origin,
                                 ((variableKeyword&) keyword).variable);
      }
      if (kType & keywordType::function) {
        return new functionToken(identifier.origin,
                                 ((functionKeyword&) keyword).function);
      }
      // keywordType::type
      return new typeToken(identifier.origin,
                           ((typeKeyword&) keyword).type_);
    }

    void tokenContext_t::debugPrint() {
      const int start = tokenIndices[tp.start];
      const int end = tokenIndices[tp.end - 1];

      for (int i = start; i < end; ++i) {
        token_t &token = *(tokens[i]);

        if (token.type() & skippableTokenTypes) {
          io::stdout << '[' << token << "] (SKIPPED)\n";
        } else {
          io::stdout << '[' << token << "]\n";
        }
      }
    }
  }
}
