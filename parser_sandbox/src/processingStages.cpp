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

#include "token.hpp"
#include "processingStages.hpp"

namespace occa {
  namespace lang {
    //---[ Newlines ]-------------------
    newlineTokenFilter::newlineTokenFilter() {}

    tokenMap& newlineTokenFilter::clone_() const {
      return *(new newlineTokenFilter());
    }

    bool newlineTokenFilter::isValid(token_t * const &token) {
      return !(token->type() & tokenType::newline);
    }
    //==================================

    //---[ Strings ]--------------------
    stringTokenMerger::stringTokenMerger() {}

    stringTokenMerger::stringTokenMerger(const stringTokenMerger &other) :
      tokenCacheMap(other) {}

    tokenMap& stringTokenMerger::clone_() const {
      return *(new stringTokenMerger(*this));
    }

    void stringTokenMerger::fetchNext() {
      token_t *token = NULL;

      *(this->input) >> token;

      // Not a string token
      if (!(token->type() & tokenType::string)) {
        pushOutput(token);
        return;
      }

      token_t *nextToken = NULL;
      stringToken &strToken = token->to<stringToken>();
      while (!inputIsEmpty()) {
        nextToken = NULL;

        // Merge until no stringToken appears
        *(this->input) >> nextToken;

        if (!(nextToken->type() & tokenType::string)) {
          break;
        }

        strToken.append(nextToken->to<stringToken>());
        delete nextToken;
        nextToken = NULL;
        // Can't merge strings with udfs in one token
        if (strToken.udf.size()) {
          break;
        }
      }

      pushOutput(&strToken);
      if (nextToken) {
        pushOutput(nextToken);
      }
    }
    //==================================

    //---[ Unknown ]--------------------
    unknownTokenFilter::unknownTokenFilter(const bool printError_) :
      printError(printError_) {}

    tokenMap& unknownTokenFilter::clone_() const {
      return *(new unknownTokenFilter(printError));
    }

    void unknownTokenFilter::setPrintError(const bool printError_) {
      printError = printError_;
    }

    bool unknownTokenFilter::isValid(token_t * const &token) {
      if (!(token->type() & tokenType::unknown)) {
        return true;
      }
      if (printError) {
        token->printError("Unknown symbol");
        delete token;
      }
      return false;
    }
    //==================================
  }
}
