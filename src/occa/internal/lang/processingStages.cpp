#include <occa/internal/lang/token.hpp>
#include <occa/internal/lang/processingStages.hpp>

namespace occa {
  namespace lang {
    //---[ Newlines ]-------------------
    newlineTokenFilter::newlineTokenFilter() {}

    tokenMap& newlineTokenFilter::clone_() const {
      return *(new newlineTokenFilter());
    }

    bool newlineTokenFilter::isValid(token_t * const &token) {
      if (token->type() & tokenType::newline) {
        delete token;
        return false;
      }
      return true;
    }
    //==================================

    //---[ Strings ]--------------------
    stringTokenMerger::stringTokenMerger() {}

    stringTokenMerger::stringTokenMerger(const stringTokenMerger &other) :
      tokenOutputCacheMap(other) {}

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

      tokenVector stack;
      stringToken &strToken = token->to<stringToken>();
      while (!inputIsEmpty()) {
        token_t *nextToken = NULL;
        *(this->input) >> nextToken;
        if (!nextToken) {
          break;
        }

        int tType = nextToken->type();
        if (!(tType & (tokenType::newline |
                       tokenType::string))) {
          stack.push_back(nextToken);
          break;
        }

        // Strings can be separated by spaces or newlines
        if (tType & tokenType::newline) {
          stack.push_back(nextToken);
          continue;
        }
        // Free all of the newline tokens
        freeTokenVector(stack);

        strToken.append(nextToken->to<stringToken>());
        // Can't merge strings with udfs in one token
        if (strToken.udf.size()) {
          break;
        }
      }

      pushOutput(&strToken);
      const int stackSize = (int) stack.size();
      for (int i = 0; i < stackSize; ++i) {
        pushOutput(stack[i]);
      }
    }
    //==================================

    //---[ Extern ]---------------------
    externTokenMerger::externTokenMerger() {}

    externTokenMerger::externTokenMerger(const externTokenMerger &other) :
      tokenCacheMap(other) {}

    tokenMap& externTokenMerger::clone_() const {
      return *(new externTokenMerger(*this));
    }

    void externTokenMerger::fetchNext() {
      token_t *token = NULL;
      getNextInput(token);

      if ((token->type() != tokenType::identifier)
          || (((identifierToken*) token)->value != "extern")
          || inputIsEmpty()) {
        pushOutput(token);
        return;
      }

      token_t *nextToken = NULL;
      getNextInput(nextToken);

      if (nextToken->type() != tokenType::string) {
        pushOutput(token);
        pushInput(nextToken);
        return;
      }

      const std::string &value = ((stringToken*) nextToken)->value;
      const bool isC   = (value == "C");
      const bool isCpp = !isC && (value == "C++");
      if (isC || isCpp) {
        pushOutput(
          new identifierToken(token->origin,
                              "extern \"" + value + "\"")
        );
        delete token;
        delete nextToken;
      } else {
        pushOutput(token);
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
