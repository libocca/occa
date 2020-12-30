#include <occa/internal/lang/token/token.hpp>
#include <occa/internal/lang/token/operatorToken.hpp>
#include <occa/internal/lang/type.hpp>

namespace occa {
  namespace lang {
    namespace charcodes {
      const char whitespace[]          = " \n\t\r\v\f";
      const char whitespaceNoNewline[] = " \t\r\v\f";

      const char alpha[] =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

      const char number[] =
        "0123456789";

      const char alphanumber[] =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789";

      const char identifierStart[] =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "_";

      const char identifier[] =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "_";
    }

    namespace encodingType {
      const int none = 0;
      const int R    = (1 << 0);
      const int u8   = (1 << 1);
      const int u    = (1 << 2);
      const int U    = (1 << 3);
      const int L    = (1 << 4);
      const int ux   = (u8 | u | U | L);
      const int bits = 5;
    }

    namespace tokenType {
      const int none          = (1 << 0);
      const int unknown       = (1 << 1);

      const int newline       = (1 << 2);

      const int systemHeader  = (1 << 3);
      const int header        = (1 << 4);

      const int directive     = (1 << 5);
      const int pragma        = (1 << 6);
      const int comment       = (1 << 7);

      const int identifier    = (1 << 8);

      const int qualifier     = (1 << 9);
      const int type          = (1 << 10);
      const int vartype       = (1 << 11);
      const int variable      = (1 << 12);
      const int function      = (1 << 13);

      const int primitive     = (1 << 14);
      const int op            = (1 << 15);

      const int char_         = (1 << 16);
      const int string        = (1 << 17);
      const int withUDF       = (1 << 18);
      const int withEncoding  = ((encodingType::ux |
                                  encodingType::R) << 19);
      const int encodingShift = 19;

      int getEncoding(const int tokenType) {
        return ((tokenType & withEncoding) >> encodingShift);
      }

      int mergeEncodings(const int encoding1, const int encoding2) {
        int rawEncoding = ((encoding1 | encoding2) & encodingType::R);
        const int encoding1_ = (encoding1 & encodingType::ux);
        const int encoding2_ = (encoding2 & encodingType::ux);
        if (encoding1_ > encoding2_) {
          return (encoding1_ | rawEncoding);
        }
        return (encoding2_ | rawEncoding);
      }
    }

    token_t::token_t(const fileOrigin &origin_) :
      origin(origin_) {}

    token_t::~token_t() {}

    int token_t::safeType(token_t *token) {
      return (token
              ? token->type()
              : tokenType::none);
    }

    opType_t token_t::safeOperatorType(token_t *token) {
      if (!(token_t::safeType(token) & tokenType::op)) {
        return operatorType::none;
      }
      return token->to<operatorToken>().getOpType();
    }

    token_t* token_t::clone(const token_t *token) {
      if (token) {
        return token->clone();
      }
      return NULL;
    }

    opType_t token_t::getOpType() {
      if (type() != tokenType::op) {
        return operatorType::none;
      }
      return to<operatorToken>().opType();
    }

    void token_t::printWarning(const std::string &message) const {
      origin.printWarning(message);
    }

    void token_t::printError(const std::string &message) const {
      origin.printError(message);
    }

    std::string token_t::str() const {
      std::stringstream ss;
      io::output out(ss);
      print(out);
      return ss.str();
    }

    void token_t::debugPrint() const {
      io::stderr << '[';
      print(io::stderr);
      io::stderr << "]\n";
    }

    io::output& operator << (io::output &out,
                             token_t &token) {
      token.print(out);
      return out;
    }

    //---[ Helper Methods ]-------------
    void freeTokenVector(tokenVector &lineTokens) {
      const int tokens = (int) lineTokens.size();
      for (int i = 0; i < tokens; ++i) {
        delete lineTokens[i];
      }
      lineTokens.clear();
    }

    std::string stringifyTokens(tokenVector &tokens,
                                const bool addSpaces) {
      std::stringstream ss;
      io::output out(ss);

      const int tokenCount = (int) tokens.size();
      for (int i = 0; i < tokenCount; ++i) {
        tokens[i]->print(out);
        // We don't add spaces between adjacent tokens
        // For example, .. would normaly turn to ". ."
        if (addSpaces              &&
            (i < (tokenCount - 1)) &&
            (tokens[i]->origin.distanceTo(tokens[i + 1]->origin))) {
          ss << ' ';
        }
      }
      return ss.str();
    }
    //==================================
  }
}
