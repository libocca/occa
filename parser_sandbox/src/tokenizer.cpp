#include "tokenizer.hpp"
#include "occa/tools/io.hpp"
#include "occa/par/tls.hpp"

namespace occa {
  namespace lang {
    namespace charcodes {
      const char whitespace[]      = " \t\r\n\v\f";
      const char alpha[]           = ("abcdefghijklmnopqrstuvwxyz"
                                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
      const char number[]          = "0123456789";
      const char alphanumber[]     = ("abcdefghijklmnopqrstuvwxyz"
                                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                      "0123456789");
      const char identifierStart[] = ("abcdefghijklmnopqrstuvwxyz"
                                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                      "_");
      const char identifier[]      = ("abcdefghijklmnopqrstuvwxyz"
                                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                      "0123456789"
                                      "_");
      const char operators[]       = "!#%&()*+,-./:;<=>?[]^{|}~";
    }

    operatorTrie& getOperators() {
      static tls<operatorTrie> operators_;
      operatorTrie &operators = operators_.value();
      if (!operators.isEmpty()) {
        return operators;
      }
      operators.add("!"  , &op::not_);
      operators.add("~"  , &op::tilde);
      operators.add("++" , &op::leftIncrement);
      operators.add("--" , &op::leftDecrement);
      operators.add("+"  , &op::add);
      operators.add("-"  , &op::sub);
      operators.add("*"  , &op::mult);
      operators.add("/"  , &op::div);
      operators.add("%"  , &op::mod);
      operators.add("<"  , &op::lessThan);
      operators.add("<=" , &op::lessThanEq);
      operators.add("==" , &op::equal);
      operators.add("!=" , &op::notEqual);
      operators.add(">"  , &op::greaterThan);
      operators.add(">=" , &op::greaterThanEq);
      operators.add("&&" , &op::and_);
      operators.add("||" , &op::or_);
      operators.add("&"  , &op::bitAnd);
      operators.add("|"  , &op::bitOr);
      operators.add("^"  , &op::xor_);
      operators.add("<<" , &op::leftShift);
      operators.add(">>" , &op::rightShift);
      operators.add("="  , &op::assign);
      operators.add("+=" , &op::addEq);
      operators.add("-=" , &op::subEq);
      operators.add("*=" , &op::multEq);
      operators.add("/=" , &op::divEq);
      operators.add("%=" , &op::modEq);
      operators.add("&=" , &op::andEq);
      operators.add("|=" , &op::orEq);
      operators.add("^=" , &op::xorEq);
      operators.add("<<=", &op::leftShiftEq);
      operators.add(">>=", &op::rightShiftEq);
      operators.add(","  , &op::comma);
      operators.add("::" , &op::scope);
      operators.add("."  , &op::dot);
      operators.add(".*" , &op::dotStar);
      operators.add("->" , &op::arrow);
      operators.add("->*", &op::arrowStar);
      operators.add("?"  , &op::ternary);
      operators.add(":"  , &op::colon);
      operators.add("{"  , &op::braceStart);
      operators.add("}"  , &op::braceEnd);
      operators.add("["  , &op::bracketStart);
      operators.add("]"  , &op::bracketEnd);
      operators.add("("  , &op::parenthesesStart);
      operators.add(")"  , &op::parenthesesEnd);
      operators.add("#"  , &op::hash);
      operators.add("##" , &op::hashhash);
      operators.add(";"  , &op::semicolon);
      operators.add("...", &op::ellipsis);
      return operators;
    }

    //---[ Tokens ]---------------------
    token_t::token_t(fileOrigin *origin_) :
      origin(origin_) {}

    token_t::~token_t() {}

    identifierToken::identifierToken(fileOrigin *origin_,
                                     const std::string &value_) :
      token_t(origin_),
      value(value_) {}

    identifierToken::~identifierToken() {}

    int identifierToken::type() const {
      return tokenType::identifier;
    }

    void identifierToken::print(printer &pout) const {
      pout << value;
    }

    primitiveToken::primitiveToken(fileOrigin *origin_,
                                   const primitive &value_) :
      token_t(origin_),
      value(value_) {}

    primitiveToken::~primitiveToken() {}

    int primitiveToken::type() const {
      return tokenType::primitive;
    }

    void primitiveToken::print(printer &pout) const {
      pout << value;
    }

    operatorToken::operatorToken(fileOrigin *origin_,
                                 const operator_t &op_) :
      token_t(origin_),
      op(op_) {}

    operatorToken::~operatorToken() {}

    int operatorToken::type() const {
      return tokenType::op;
    }

    void operatorToken::print(printer &pout) const {
      op.print(pout);
    }

    charToken::charToken(fileOrigin *origin_,
                         int uType_,
                         const std::string &value_,
                         const std::string &udf_) :
      token_t(origin_),
      uType(uType_),
      value(value_),
      udf(udf_) {}

    charToken::~charToken() {}

    int charToken::type() const {
      return tokenType::char_;
    }

    void charToken::print(printer &pout) const {
      if (uType & encodingType::ux) {
        if (uType & encodingType::u8) {
          pout << "u8";
        } else if (uType & encodingType::u) {
          pout << 'u';
        } else if (uType & encodingType::U) {
          pout << 'U';
        } else if (uType & encodingType::L) {
          pout << 'L';
        }
      }
      if (uType & encodingType::R) {
        pout << 'R';
      }
      pout << '\'' << value << '\'' << udf;
    }

    stringToken::~stringToken() {}

    stringToken::stringToken(fileOrigin *origin_,
                             int uType_,
                             const std::string &value_,
                             const std::string &udf_) :
      token_t(origin_),
      uType(uType_),
      value(value_),
      udf(udf_) {}

    int stringToken::type() const {
      return tokenType::string;
    }

    void stringToken::print(printer &pout) const {
      if (uType & encodingType::ux) {
        if (uType & encodingType::u8) {
          pout << "u8";
        } else if (uType & encodingType::u) {
          pout << 'u';
        } else if (uType & encodingType::U) {
          pout << 'U';
        } else if (uType & encodingType::L) {
          pout << 'L';
        }
      }
      if (uType & encodingType::R) {
        pout << 'R';
      }
      pout << '"' << value << '"' << udf;
    }

    headerToken::headerToken(fileOrigin *origin_,
                             const bool systemHeader_,
                             const std::string &value_) :
      token_t(origin_),
      systemHeader(systemHeader_),
      value(value_) {}

    headerToken::~headerToken() {}

    int headerToken::type() const {
      return tokenType::header;
    }

    void headerToken::print(printer &pout) const {
      if (systemHeader) {
        pout << '<' << value << '>';
      } else {
        pout << '"' << value << '"';
      }
    }
    //==================================

    int getEncodingType(const std::string &str) {
      int encoding      = 0;
      int encodingCount = 0;
      const char *c = str.c_str();
      while (*c) {
        int newEncoding = 0;
        switch (*c) {
        case 'u': {
          if (c[1] == '8') {
            newEncoding = encodingType::u8;
            ++c;
          } else {
            newEncoding = encodingType::u;
          }
          break;
        }
        case 'U':
          newEncoding = encodingType::U; break;
        case 'L':
          newEncoding = encodingType::L; break;
        case 'R':
          newEncoding = encodingType::R; break;
        }
        if (!newEncoding ||
            (newEncoding & encoding)) {
          return encodingType::none;
        }
        encoding |= newEncoding;
        ++encodingCount;
        ++c;
      }
      if ((encodingCount == 1) ||
          ((encodingCount == 2) && (encoding & encodingType::R))) {
        return encoding;
      }
      return encodingType::none;
    }

    int getCharacterEncoding(const std::string &str) {
      const int encoding = getEncodingType(str);
      if (encoding &&
          !(encoding & encodingType::u8)) {
        return encoding;
      }
      return encodingType::none;
    }

    int getStringEncoding(const std::string &str) {
      return getEncodingType(str);
    }

    //---[ Character Stream ]-----------
    charStringInfo::charStringInfo(const char *root_) :
      root(root_),
      pos(root_),
      newlinesPassed(0) {}

    charStringInfo::charStringInfo(const charStringInfo &other) :
      root(other.root),
      pos(other.pos),
      newlinesPassed(other.newlinesPassed) {}

    charStream::charStream(const char *root_) :
      root(root_),
      pos(root_),
      newlinesPassed(0) {}

    charStream::charStream(const charStream &stream) :
      root(stream.root),
      pos(stream.pos),
      newlinesPassed(stream.newlinesPassed),
      stack(stream.stack) {}

    const char * charStream::getPosition() {
      return pos;
    }

    void charStream::setPosition(const char * pos_) {
      pos = pos_;
    }

    void charStream::clear() {
      newlinesPassed = 0;
    }

    void charStream::push() {
      charStringInfo info(root);
      info.pos = pos;
      info.newlinesPassed = info.newlinesPassed;
      stack.push_back(info);

      newlinesPassed = 0;
    }

    void charStream::pop(const bool rewind) {
      if (stack.size() > 0) {
        if (rewind) {
          charStringInfo info = stack.back();
          root = info.root;
          pos = info.pos;
          newlinesPassed = info.newlinesPassed;
        }
        stack.pop_back();
      }
    }

    std::string charStream::str() {
      if (stack.size() == 0) {
        return "";
      }
      charStringInfo last = stack.back();
      return std::string(last.pos, pos - last.pos);
    }

    void charStream::skipTo(const char delimiter) {
      while (*pos != '\0') {
        if (*pos == delimiter) {
          return;
        }
        if (*pos == '\n') {
          ++newlinesPassed;
        }
        ++pos;
      }
    }

    void charStream::skipTo(const char delimiter,
                            const char escapeChar) {
      while (*pos != '\0') {
        if (*pos == escapeChar) {
          pos += 1 + (pos[1] != '\0');
          continue;
        }
        if (*pos == delimiter) {
          return;
        }
        if (*pos == '\n') {
          ++newlinesPassed;
        }
        ++pos;
      }
    }

    void charStream::skipTo(const char *delimiters) {
      while (*pos != '\0') {
        if (lex::charIsIn(*pos, delimiters)) {
          return;
        }
        if (*pos == '\n') {
          ++newlinesPassed;
        }
        ++pos;
      }
    }

    void charStream::skipTo(const char *delimiters,
                            const char escapeChar) {
      while (*pos != '\0') {
        if (*pos == escapeChar) {
          pos += 1 + (pos[1] != '\0');
          continue;
        }
        if (lex::charIsIn(*pos, delimiters)) {
          return;
        }
        if (*pos == '\n') {
          ++newlinesPassed;
        }
        ++pos;
      }
    }

    void charStream::skipFrom(const char *delimiters) {
      while (*pos != '\0') {
        if (lex::charIsIn(*pos, delimiters)) {
          ++pos;
          continue;
        }
        if (*pos == '\n') {
          ++newlinesPassed;
        }
        return;
      }
    }

    void charStream::skipFrom(const char *delimiters, const char escapeChar) {
      while (*pos != '\0') {
        if (*pos == escapeChar) {
          pos += 1 + (pos[1] != '\0');
          continue;
        }
        if (lex::charIsIn(*pos, delimiters)) {
          ++pos;
          continue;
        }
        if (*pos == '\n') {
          ++newlinesPassed;
        }
        return;
      }
    }

    void charStream::skipWhitespace() {
      skipFrom(charcodes::whitespace);
    }

    int charStream::peek() {
      int type = shallowPeek();
      if (type == tokenType::identifier) {
        return peekForIdentifier();
      }
      return type;
    }

    int charStream::shallowPeek() {
      const char c = *pos;
      if (c == '\0') {
        return tokenType::none;
      }
      if (lex::charIsIn(c, charcodes::identifierStart)) {
        return tokenType::identifier;
      }
      if (lex::charIsIn(c, charcodes::operators)) {
        return tokenType::op;
      }
      const char *pos2 = pos;
      if (primitive::load(pos2).type != primitiveType::none) {
        return tokenType::primitive;
      }
      if (c == '"') {
        return tokenType::string;
      }
      if (c == '\'') {
        return tokenType::char_;
      }
      // TODO: Print proper error
      OCCA_FORCE_ERROR("Could not find token type");
      return tokenType::none;
    }

    int charStream::peekForIdentifier() {
      push();
      // TODO: Fix
      // skipIdentifier();
      const std::string identifier = str();
      int type = shallowPeek();
      pop();

      // [u8]"foo" or [u8]'foo'
      if ((type & tokenType::char_) &&
          getCharacterEncoding(identifier)) {
        // u8["foo"]
        return (tokenType::withUType
                | tokenType::string);
      }
      if ((type & tokenType::string) &&
          getStringEncoding(identifier)) {
        // u8['foo']
        return (tokenType::withUType
                | tokenType::char_);
      }
      return tokenType::identifier;
    }

    int charStream::peekForHeader() {
      int type = shallowPeek();
      if (type & tokenType::op) {
        push();
        // TODO: Fix
        // skipOperator();
        const std::string op = str();
        pop();
        if (op == "<") {
          return tokenType::systemHeader;
        }
      } else if (type & tokenType::string) {
        return tokenType::header;
      }
      return tokenType::none;
    }

    token_t* charStream::getToken() {
      int type = peek();
      if (type & tokenType::identifier) {
        return getIdentifierToken();
      }
      if (type & tokenType::header) {
        return getHeaderToken();
      }
      if (type & tokenType::primitive) {
        return getPrimitiveToken();
      }
      if (type & tokenType::op) {
        return getOperatorToken();
      }
      if (type & tokenType::char_) {
        return getCharToken();
      }
      if (type & tokenType::string) {
        return getStringToken();
      }
      return NULL;
    }

    token_t* charStream::getIdentifierToken() {
      // TODO: Print proper error
      OCCA_ERROR("Not able to parse identifier",
                 lex::charIsIn(*pos, charcodes::identifierStart));
      push();
      ++pos;
      skipFrom(charcodes::identifier);
      return new identifierToken(NULL, str());
      pop();
    }

    token_t* charStream::getPrimitiveToken() {
      primitive value = primitive::load(pos);
      // TODO: Print proper error
      OCCA_ERROR("Not able to parse primitive",
                 !value.isNaN());
      return new primitiveToken(NULL, value);
    }

    token_t* charStream::getOperatorToken() {
      operatorTrie &operators = getOperators();
      operatorTrie::result_t result = operators.getLongest(pos);
      // TODO: Print proper error
      OCCA_ERROR("Not able to parse operator",
                 result.success());
      return new operatorToken(NULL, *(result.value()));
    }

    token_t* charStream::getStringToken() {
      if (*pos != '"') {
        // TODO: Print proper error
        OCCA_FORCE_ERROR("Not able to parse string");
      }
      ++pos;
      push();
      skipTo('"', '\\');
      token_t *token = new stringToken(NULL,
                                       encodingType::u8,
                                       str(),
                                       "_km");
      pop(false);
      ++pos;
      return token;
    }

    token_t* charStream::getCharToken() {
      if (*pos != '\'') {
        // TODO: Print proper error
        OCCA_FORCE_ERROR("Not able to parse string");
      }
      ++pos;
      push();
      skipTo('\'', '\\');
      token_t *token = new charToken(NULL,
                                     encodingType::u,
                                     str(),
                                     "_km");
      pop(false);
      ++pos;
      return token;
    }

    token_t* charStream::getHeaderToken() {
      int type = shallowPeek();
      if (type & tokenType::op) {
        ++pos;
        push();
        token_t *token = new headerToken(NULL,
                                         true,
                                         str());
        skipTo('>');
        ++pos;
        pop(false);
        return token;
      }
      if (type & tokenType::string) {
        stringToken &token = *((stringToken*) getStringToken());
        std::string value = token.value;
        delete &token;
        return new headerToken(NULL, false, value);
      }
      // TODO: Print proper error
      OCCA_FORCE_ERROR("Not able to parse header");
      return new headerToken(NULL, false, "");
    }
    //==================================

    //---[ Tokenizer ]------------------

    //==================================
  }
}
