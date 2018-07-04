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
#include <occa/lang/expression.hpp>
#include <occa/lang/token.hpp>
#include <occa/lang/tokenContext.hpp>

namespace occa {
  namespace lang {
    tokenRange::tokenRange() :
      start(0),
      end(0) {}

    tokenRange::tokenRange(const int start_,
                           const int end_) :
      start(start_),
      end(end_) {}

    tokenContext::tokenContext() {}

    tokenContext::~tokenContext() {
      clear();
    }

    void tokenContext::clear() {
      tp.start = 0;
      tp.end   = 0;

      hasError = false;

      const int tokenCount = (int) tokens.size();
      for (int i = 0; i < tokenCount; ++i) {
        delete tokens[i];
      }
      tokens.clear();
      pairs.clear();
      semicolons.clear();
      stack.clear();
    }

    void tokenContext::setup() {
      tp.start = 0;
      tp.end   = (int) tokens.size();

      findPairs();
      findSemicolons();
    }

    void tokenContext::findPairs() {
      intVector pairStack;

      const int tokenCount = (int) tokens.size();
      for (int i = 0; i < tokenCount; ++i) {
        token_t *token = tokens[i];
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
          std::stringstream ss;
          ss << "Could not find an opening '"
             << pairEndOp.pairStr
             << '\'';
          token->printError(ss.str());
          hasError = true;
          return;
        }

        // Make sure we close the pair
        const int pairIndex = pairStack.back();
        pairStack.pop_back();
        pairOperator_t &pairStartOp =
          *((pairOperator_t*) tokens[pairIndex]->to<operatorToken>().op);

        if (pairStartOp.opType != (pairEndOp.opType >> 1)) {
          std::stringstream ss;
          ss << "Could not find a closing '"
             << pairStartOp.pairStr
             << '\'';
          tokens[pairIndex]->printError(ss.str());
          hasError = true;
          return;
        }

        // Store pairs
        pairs[pairIndex] = i;
      }
    }

    void tokenContext::findSemicolons() {
      const int tokenCount = (int) tokens.size();
      for (int i = 0; i < tokenCount; ++i) {
        token_t *token = tokens[i];
        opType_t opType = token->getOpType();
        if (opType & operatorType::semicolon) {
          semicolons.push_back(i);
        }
      }
    }

    bool tokenContext::indexInRange(const int index) const {
      return ((index >= 0) && ((tp.start + index) < tp.end));
    }

    void tokenContext::set(const int start) {
      if (indexInRange(start)) {
        tp.start += start;
      } else {
        tp.start = tp.end;
      }
    }

    void tokenContext::set(const int start,
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

    void tokenContext::set(const tokenRange &range) {
      set(range.start, range.end);
    }

    void tokenContext::push() {
      stack.push_back(tp);
    }

    void tokenContext::push(const int start) {
      stack.push_back(tp);
      set(start);
    }

    void tokenContext::push(const int start,
                            const int end) {
      stack.push_back(tp);
      set(start, end);
    }

    void tokenContext::push(const tokenRange &range) {
      push(range.start, range.end);
    }

    void tokenContext::pushPairRange(const int pairStart) {
      const int pairEnd = getClosingPair(pairStart);
      if (pairEnd >= 0) {
        push(pairStart + 1, pairEnd);
      } else {
        OCCA_FORCE_ERROR("Trying to push a pair range without a pair");
      }
    }

    tokenRange tokenContext::pop() {
      OCCA_ERROR("Unable to call tokenContext::pop",
                 stack.size());

      tokenRange prev = tp;
      tp = stack.back();
      stack.pop_back();

      // Return a relative tokenRange
      const int prevStart = prev.start - tp.start;
      return tokenRange(prevStart,
                        prevStart + (prev.end - prev.start));
    }

    void tokenContext::popAndSkip() {
      set(pop().end + 1);
    }

    int tokenContext::position() const {
      return tp.start;
    }

    int tokenContext::size() const {
      return (tp.end - tp.start);
    }

    token_t* tokenContext::operator [] (const int index) {
      if (!indexInRange(index)) {
        return NULL;
      }
      return tokens[tp.start + index];
    }

    void tokenContext::setToken(const int index,
                                token_t *value) {
      if (!indexInRange(index)) {
        return;
      }
      const int pos = tp.start + index;
      if (tokens[pos] != value) {
        delete tokens[pos];
        tokens[pos] = value;
      }
    }

    token_t* tokenContext::end() {
      if (indexInRange(tp.end - tp.start - 1)) {
        return tokens[tp.end - 1];
      }
      return NULL;
    }

    token_t* tokenContext::getPrintToken(const bool atEnd) {
      if (tokens.size() == 0) {
        return NULL;
      }
      const int start = tp.start;
      if (atEnd) {
        tp.start = tp.end;
      }
      int offset = 0;
      if (!indexInRange(offset) &&
          (0 < tp.start)) {
        offset = -1;
      }
      token_t *token = tokens[tp.start + offset];
      if (atEnd) {
        tp.start = start;
      }
      return token;
    }

    void tokenContext::printWarning(const std::string &message) {
      token_t *token = getPrintToken(false);
      if (!token) {
        occa::printWarning(std::cerr, "[No Token] " + message);
      } else {
        token->printWarning(message);
      }
    }

    void tokenContext::printWarningAtEnd(const std::string &message) {
      token_t *token = getPrintToken(true);
      if (!token) {
        occa::printWarning(std::cerr, "[No Token] " + message);
      } else {
        token->printWarning(message);
      }
    }

    void tokenContext::printError(const std::string &message) {
      token_t *token = getPrintToken(false);
      if (!token) {
        occa::printError(std::cerr, "[No Token] " + message);
      } else {
        token->printError(message);
      }
    }

    void tokenContext::printErrorAtEnd(const std::string &message) {
      token_t *token = getPrintToken(true);
      if (!token) {
        occa::printError(std::cerr, "[No Token] " + message);
      } else {
        token->printError(message);
      }
    }

    void tokenContext::getTokens(tokenVector &tokens_) {
      tokens_.clear();
      tokens_.reserve(tp.end - tp.start);
      for (int i = tp.start; i < tp.end; ++i) {
        tokens_.push_back(tokens[i]);
      }
    }

    void tokenContext::getAndCloneTokens(tokenVector &tokens_) {
      tokens_.clear();
      tokens_.reserve(tp.end - tp.start);
      for (int i = tp.start; i < tp.end; ++i) {
        tokens_.push_back(tokens[i]->clone());
      }
    }

    int tokenContext::getClosingPair(const int index) {
      if (!indexInRange(index)) {
        return -1;
      }

      intIntMap::iterator it = pairs.find(tp.start + index);
      if (it != pairs.end()) {
        return (it->second - tp.start);
      }
      return -1;
    }

    token_t* tokenContext::getClosingPairToken(const int index) {
      const int endIndex = getClosingPair(index);
      if (endIndex >= 0) {
        return tokens[tp.start + endIndex];
      }
      return NULL;
    }

    int tokenContext::getNextOperator(const opType_t &opType) {
      for (int pos = tp.start; pos < tp.end; ++pos) {
        token_t *token = tokens[pos];
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

    void tokenContext::debugPrint() {
      for (int i = tp.start; i < tp.end; ++i) {
        std::cout << '[' << *tokens[i] << "]\n";
      }
    }
  }
}
