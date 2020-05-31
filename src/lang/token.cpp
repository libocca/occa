#include <occa/lang/token.hpp>
#include <occa/lang/type.hpp>
#include <occa/lang/variable.hpp>

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

    //---[ Unknown ]--------------------
    unknownToken::unknownToken(const fileOrigin &origin_) :
      token_t(origin_) {}

    unknownToken::~unknownToken() {}

    int unknownToken::type() const {
      return tokenType::unknown;
    }

    token_t* unknownToken::clone() const {
      return new unknownToken(origin);
    }

    void unknownToken::print(io::output &out) const {
      out << origin.position.start[0];
    }
    //==================================

    //---[ Newline ]--------------------
    newlineToken::newlineToken(const fileOrigin &origin_) :
      token_t(origin_) {}

    newlineToken::~newlineToken() {}

    int newlineToken::type() const {
      return tokenType::newline;
    }

    token_t* newlineToken::clone() const {
      return new newlineToken(origin);
    }

    void newlineToken::print(io::output &out) const {
      out << '\n';
    }
    //==================================

    //---[ Directive ]------------------
    directiveToken::directiveToken(const fileOrigin &origin_,
                                   const std::string &value_) :
      token_t(origin_),
      value(value_) {}

    directiveToken::~directiveToken() {}

    int directiveToken::type() const {
      return tokenType::directive;
    }

    token_t* directiveToken::clone() const {
      return new directiveToken(origin, value);
    }

    void directiveToken::print(io::output &out) const {
      out << '#' << value << '\n';
    }
    //==================================

    //---[ Pragma ]---------------------
    pragmaToken::pragmaToken(const fileOrigin &origin_,
                             const std::string &value_) :
      token_t(origin_),
      value(value_) {}

    pragmaToken::~pragmaToken() {}

    int pragmaToken::type() const {
      return tokenType::pragma;
    }

    token_t* pragmaToken::clone() const {
      return new pragmaToken(origin, value);
    }

    void pragmaToken::print(io::output &out) const {
      out << "#pragma " << value << '\n';
    }
    //==================================

    //---[ Comment ]-----------------
    commentToken::commentToken(const fileOrigin &origin_,
                               const std::string &value_) :
        token_t(origin_),
        value(value_) {}

    commentToken::~commentToken() {}

    int commentToken::type() const {
      return tokenType::comment;
    }

    token_t* commentToken::clone() const {
      return new commentToken(origin, value);
    }

    void commentToken::print(io::output &out) const {
      out << value;
    }
    //==================================

    //---[ Identifier ]-----------------
    identifierToken::identifierToken(const fileOrigin &origin_,
                                     const std::string &value_) :
      token_t(origin_),
      value(value_) {}

    identifierToken::~identifierToken() {}

    int identifierToken::type() const {
      return tokenType::identifier;
    }

    token_t* identifierToken::clone() const {
      return new identifierToken(origin, value);
    }

    void identifierToken::print(io::output &out) const {
      out << value;
    }
    //==================================

    //---[ Qualifier ]------------------
    qualifierToken::qualifierToken(const fileOrigin &origin_,
                                   const qualifier_t &qualifier_) :
      token_t(origin_),
      qualifier(qualifier_) {}

    qualifierToken::~qualifierToken() {}

    int qualifierToken::type() const {
      return tokenType::qualifier;
    }

    token_t* qualifierToken::clone() const {
      return new qualifierToken(origin, qualifier);
    }

    void qualifierToken::print(io::output &out) const {
      out << qualifier.name;
    }
    //==================================

    //---[ Type ]-----------------------
    typeToken::typeToken(const fileOrigin &origin_,
                         type_t &type_) :
      token_t(origin_),
      value(type_) {}

    typeToken::~typeToken() {}

    int typeToken::type() const {
      return tokenType::type;
    }

    token_t* typeToken::clone() const {
      return new typeToken(origin, value);
    }

    void typeToken::print(io::output &out) const {
      out << value.name();
    }
    //==================================

    //---[ Vartype ]--------------------
    vartypeToken::vartypeToken(const fileOrigin &origin_,
                               const vartype_t &value_) :
      token_t(origin_),
      value(value_) {}

    vartypeToken::~vartypeToken() {}

    int vartypeToken::type() const {
      return tokenType::vartype;
    }

    token_t* vartypeToken::clone() const {
      return new vartypeToken(origin, value);
    }

    void vartypeToken::print(io::output &out) const {
      out << value;
    }
    //==================================

    //---[ Variable ]-------------------
    variableToken::variableToken(const fileOrigin &origin_,
                                 variable_t &variable) :
      token_t(origin_),
      value(variable) {}

    variableToken::~variableToken() {}

    int variableToken::type() const {
      return tokenType::variable;
    }

    token_t* variableToken::clone() const {
      return new variableToken(origin, value);
    }

    void variableToken::print(io::output &out) const {
      out << value.name();
    }
    //==================================

    //---[ Function ]-------------------
    functionToken::functionToken(const fileOrigin &origin_,
                                 function_t &function) :
      token_t(origin_),
      value(function) {}

    functionToken::~functionToken() {}

    int functionToken::type() const {
      return tokenType::function;
    }

    token_t* functionToken::clone() const {
      return new functionToken(origin, value);
    }

    void functionToken::print(io::output &out) const {
      out << value.name();
    }
    //==================================

    //---[ Primitive ]------------------
    primitiveToken::primitiveToken(const fileOrigin &origin_,
                                   const primitive &value_,
                                   const std::string &strValue_) :
      token_t(origin_),
      value(value_),
      strValue(strValue_) {}

    primitiveToken::~primitiveToken() {}

    int primitiveToken::type() const {
      return tokenType::primitive;
    }

    token_t* primitiveToken::clone() const {
      return new primitiveToken(origin, value, strValue);
    }

    void primitiveToken::print(io::output &out) const {
      out << strValue;
    }
    //==================================

    //---[ Operator ]-------------------
    operatorToken::operatorToken(const fileOrigin &origin_,
                                 const operator_t &op_) :
      token_t(origin_),
      op(&op_) {}

    operatorToken::~operatorToken() {}

    int operatorToken::type() const {
      return tokenType::op;
    }

    const opType_t& operatorToken::opType() const {
      return op->opType;
    }

    token_t* operatorToken::clone() const {
      return new operatorToken(origin, *op);
    }

    void operatorToken::print(io::output &out) const {
      out << op->str;
    }
    //==================================

    //---[ Char ]-----------------------
    charToken::charToken(const fileOrigin &origin_,
                         int encoding_,
                         const std::string &value_,
                         const std::string &udf_) :
      token_t(origin_),
      encoding(encoding_),
      value(value_),
      udf(udf_) {}

    charToken::~charToken() {}

    int charToken::type() const {
      return tokenType::char_;
    }

    token_t* charToken::clone() const {
      return new charToken(origin, encoding, value, udf);
    }

    void charToken::print(io::output &out) const {
      if (encoding & encodingType::u) {
        out << 'u';
      } else if (encoding & encodingType::U) {
        out << 'U';
      } else if (encoding & encodingType::L) {
        out << 'L';
      }
      out << '\'' << escape(value, '\'') << '\'' << udf;
    }
    //==================================

    //---[ String ]---------------------
    stringToken::stringToken(const fileOrigin &origin_,
                             const std::string &value_) :
      token_t(origin_),
      encoding(encodingType::none),
      value(value_),
      udf() {}

    stringToken::stringToken(const fileOrigin &origin_,
                             int encoding_,
                             const std::string &value_,
                             const std::string &udf_) :
      token_t(origin_),
      encoding(encoding_),
      value(value_),
      udf(udf_) {}

    stringToken::~stringToken() {}

    int stringToken::type() const {
      return tokenType::string;
    }

    token_t* stringToken::clone() const {
      return new stringToken(origin, encoding, value, udf);
    }

    void stringToken::append(const stringToken &token) {
      origin.position.end = token.origin.position.end;

      encoding = tokenType::mergeEncodings(encoding,
                                           token.encoding);
      value += token.value;
      udf = token.udf;
    }

    void stringToken::print(io::output &out) const {
      if (encoding & encodingType::ux) {
        if (encoding & encodingType::u8) {
          out << "u8";
        } else if (encoding & encodingType::u) {
          out << 'u';
        } else if (encoding & encodingType::U) {
          out << 'U';
        } else if (encoding & encodingType::L) {
          out << 'L';
        }
      }
      if (encoding & encodingType::R) {
        out << 'R';
      }
      out << '"' << escape(value, '"') << '"' << udf;
    }
    //==================================


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
