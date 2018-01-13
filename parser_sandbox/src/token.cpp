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
      const int none          = 0;

      const int identifier    = (1 << 0);

      const int systemHeader  = (1 << 1);
      const int header        = (3 << 1);

      const int primitive     = (1 << 3);
      const int op            = (1 << 4);

      const int char_         = (1 << 5);
      const int string        = (1 << 6);
      const int withUDF       = (1 << 7);
      const int withEncoding  = ((encodingType::ux |
                                  encodingType::R) << 8);
      const int encodingShift = 8;
    }

    token_t::token_t(const fileOrigin &origin_) :
      origin(origin_) {}

    token_t::~token_t() {}

    identifierToken::identifierToken(const fileOrigin &origin_,
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

    primitiveToken::primitiveToken(const fileOrigin &origin_,
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

    operatorToken::operatorToken(const fileOrigin &origin_,
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

    void charToken::print(printer &pout) const {
      if (encoding & encodingType::u) {
        pout << 'u';
      } else if (encoding & encodingType::U) {
        pout << 'U';
      } else if (encoding & encodingType::L) {
        pout << 'L';
      }
      pout << '\'' << value << '\'' << udf;
    }

    stringToken::~stringToken() {}

    stringToken::stringToken(const fileOrigin &origin_,
                             int encoding_,
                             const std::string &value_,
                             const std::string &udf_) :
      token_t(origin_),
      encoding(encoding_),
      value(value_),
      udf(udf_) {}

    int stringToken::type() const {
      return tokenType::string;
    }

    void stringToken::print(printer &pout) const {
      if (encoding & encodingType::ux) {
        if (encoding & encodingType::u8) {
          pout << "u8";
        } else if (encoding & encodingType::u) {
          pout << 'u';
        } else if (encoding & encodingType::U) {
          pout << 'U';
        } else if (encoding & encodingType::L) {
          pout << 'L';
        }
      }
      if (encoding & encodingType::R) {
        pout << 'R';
      }
      pout << '"' << value << '"' << udf;
    }

    headerToken::headerToken(const fileOrigin &origin_,
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
      if (!encoding ||
          (encoding & (encodingType::u8 |
                       encodingType::R))) {
        return encodingType::none;
      }
      return encoding;
    }

    int getStringEncoding(const std::string &str) {
      return getEncodingType(str);
    }

    //---[ Character Stream ]-----------
    tokenStream::tokenStream(const char *root) :
      file(NULL),
      fp(root) {}

    tokenStream::tokenStream(file_t *file_,
                             const char *root) :
      file(file_),
      fp(root) {}

    tokenStream::tokenStream(const tokenStream &stream) :
      file(stream.file),
      fp(stream.fp),
      stack(stream.stack) {}

    const char *tokenStream::getPosition() {
      return fp.pos;
    }

    void tokenStream::setPosition(const char * pos) {
      fp.pos = pos;
    }

    int tokenStream::getLine() {
      return fp.line;
    }

    void tokenStream::setLine(const int line) {
      fp.line = line;
    }

    fileOrigin tokenStream::getFileOrigin() {
      return fileOrigin(file, fp);
    }

    void tokenStream::push() {
      stack.push_back(
        filePosition(fp.line,
                     fp.lineStart,
                     fp.pos)
      );
    }

    void tokenStream::pop(const bool rewind) {
      if (stack.size() > 0) {
        if (rewind) {
          fp = stack.back();
        }
        stack.pop_back();
      }
    }

    void tokenStream::popAndRewind() {
      pop(true);
    }

    std::string tokenStream::str() {
      if (stack.size() == 0) {
        return "";
      }
      filePosition last = stack.back();
      return std::string(last.pos, fp.pos - last.pos);
    }

    void tokenStream::countSkippedLines() {
      if (stack.size() == 0) {
        return;
      }
      filePosition last = stack.back();
      const char *pos = last.pos;
      const char *end = fp.pos;
      while (pos < end) {
        if (*pos == '\\') {
          pos += 1 + (pos[1] != '\0');
          continue;
        }
        if (*pos == '\n') {
          fp.lineStart = fp.pos + 1;
          ++fp.line;
        }
        ++pos;
      }
    }

    void tokenStream::skipTo(const char delimiter) {
      while (*fp.pos != '\0') {
        if (*fp.pos == '\\') {
          fp.pos += 1 + (fp.pos[1] != '\0');
          continue;
        }
        if (*fp.pos == delimiter) {
          return;
        }
        if (*fp.pos == '\n') {
          fp.lineStart = fp.pos + 1;
          ++fp.line;
        }
        ++fp.pos;
      }
    }

    void tokenStream::skipTo(const char *delimiters) {
      while (*fp.pos != '\0') {
        if (*fp.pos == '\\') {
          fp.pos += 1 + (fp.pos[1] != '\0');
          continue;
        }
        if (lex::charIsIn(*fp.pos, delimiters)) {
          return;
        }
        if (*fp.pos == '\n') {
          fp.lineStart = fp.pos + 1;
          ++fp.line;
        }
        ++fp.pos;
      }
    }

    void tokenStream::skipFrom(const char *delimiters) {
      while (*fp.pos != '\0') {
        if (*fp.pos == '\\') {
          fp.pos += 1 + (fp.pos[1] != '\0');
          continue;
        }
        if (lex::charIsIn(*fp.pos, delimiters)) {
          ++fp.pos;
          continue;
        }
        if (*fp.pos == '\n') {
          fp.lineStart = fp.pos + 1;
          ++fp.line;
        }
        return;
      }
    }

    void tokenStream::skipWhitespace() {
      skipFrom(charcodes::whitespace);
    }

    int tokenStream::peek() {
      int type = shallowPeek();
      if (type == tokenType::identifier) {
        return peekForIdentifier();
      }
      return type;
    }

    int tokenStream::shallowPeek() {
      const char c = *fp.pos;
      if (c == '\0') {
        return tokenType::none;
      }
      if (lex::charIsIn(c, charcodes::identifierStart)) {
        return tokenType::identifier;
      }
      // Primitive must be checked before operators since
      //   it can start with + or -
      const char *pos = fp.pos;
      if (primitive::load(pos).type != primitiveType::none) {
        return tokenType::primitive;
      }
      if (lex::charIsIn(c, charcodes::operators)) {
        return tokenType::op;
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

    int tokenStream::peekForIdentifier() {
      push();
      ++fp.pos;
      skipFrom(charcodes::identifier);
      const std::string identifier = str();
      int type = shallowPeek();
      popAndRewind();

      // [u8R]"foo" or [u]'foo'
      if (type & tokenType::string) {
        const int encoding = getStringEncoding(identifier);
        if (encoding) {
          return (tokenType::string |
                  (encoding << tokenType::encodingShift));
        }
      }
      if (type & tokenType::char_) {
        const int encoding = getStringEncoding(identifier);
        if (encoding) {
          return (tokenType::char_ |
                  (encoding << tokenType::encodingShift));
        }
      }
      return tokenType::identifier;
    }

    int tokenStream::peekForHeader() {
      int type = shallowPeek();
      if (type & tokenType::op) {
        push();
        operatorTrie &operators = getOperators();
        operatorTrie::result_t result = operators.getLongest(fp.pos);
        popAndRewind();
        if (result.success() &&
            (result.value()->optype & operatorType::lessThan)) {
          return tokenType::systemHeader;
        }
      } else if (type & tokenType::string) {
        return tokenType::header;
      }
      return tokenType::none;
    }

    void tokenStream::getIdentifier(std::string &value) {
      if (!lex::charIsIn(*fp.pos, charcodes::identifierStart)) {
        return;
      }
      push();
      ++fp.pos;
      skipFrom(charcodes::identifier);
      value = str();
      pop();
    }

    void tokenStream::getString(std::string &value,
                                const int encoding) {
      if (*fp.pos != '"') {
        return;
      }
      push();
      ++fp.pos;
      push();
      if (!(encoding & encodingType::R)) {
        skipTo("\"\n");
        if (*fp.pos == '\n') {
          pop();
          popAndRewind();
          return;
        }
      } else {
        skipTo("\"");
      }
      value = str();
      pop();
      pop();
      ++fp.pos;
    }

    token_t* tokenStream::getToken() {
      skipWhitespace();
      int type = peek();
      if (type & tokenType::identifier) {
        return getIdentifierToken();
      }
      if (type & tokenType::primitive) {
        return getPrimitiveToken();
      }
      if (type & tokenType::op) {
        return getOperatorToken();
      }
      if (type & tokenType::char_) {
        return getCharToken(type >> tokenType::encodingShift);
      }
      if (type & tokenType::string) {
        return getStringToken(type >> tokenType::encodingShift);
      }
      return NULL;
    }

    token_t* tokenStream::getIdentifierToken() {
      if (!lex::charIsIn(*fp.pos, charcodes::identifierStart)) {
        // TODO: Print proper error
        // "Not able to parse identifier"
        return NULL;
      }
      std::string value;
      getIdentifier(value);
      return new identifierToken(getFileOrigin(),
                                 value);
    }

    token_t* tokenStream::getPrimitiveToken() {
      push();
      primitive value = primitive::load(fp.pos);
      if (value.isNaN()) {
        // TODO: Print proper error
        // "Not able to parse primitive"
        popAndRewind();
        return NULL;
      }
      countSkippedLines();
      pop();
      return new primitiveToken(getFileOrigin(),
                                value);
    }

    token_t* tokenStream::getOperatorToken() {
      operatorTrie &operators = getOperators();
      operatorTrie::result_t result = operators.getLongest(fp.pos);
      if (!result.success()) {
        // TODO: Print proper error
        // "Not able to parse operator"
        return NULL;
      }
      fp.pos += result.length;
      return new operatorToken(getFileOrigin(),
                               *(result.value()));
    }

    token_t* tokenStream::getStringToken(const int encoding) {
      if (encoding) {
        std::string encodingStr;
        getIdentifier(encodingStr);
      }
      if (*fp.pos != '"') {
        // TODO: Print proper error
        // "Not able to parse string"
        return NULL;
      }
      const char *start = fp.pos;
      std::string value, udf;
      getString(value, encoding);
      if (fp.pos == start) {
        // TODO: Print proper error
        // "Unable to find closing \"
        return NULL;
      }
      if (*fp.pos == '_') {
        getIdentifier(udf);
      }
      return new stringToken(getFileOrigin(),
                             encoding, value, udf);
    }

    token_t* tokenStream::getCharToken(const int encoding) {
      if (encoding) {
        std::string encodingStr;
        getIdentifier(encodingStr);
      }
      if (*fp.pos != '\'') {
        // TODO: Print proper error
        // "Not able to parse char"
        return NULL;
      }

      ++fp.pos;
      push();
      skipTo("'\n");
      if (*fp.pos == '\n') {
        // TODO: Print proper error
        // "Unable to find closing '
        popAndRewind();
        return NULL;
      }
      const std::string value = str();
      pop();

      std::string udf;
      if (*fp.pos == '_') {
        getIdentifier(udf);
      }
      return new charToken(getFileOrigin(),
                           encoding, value, udf);
    }

    token_t* tokenStream::getHeaderToken() {
      int type = shallowPeek();
      if (type & tokenType::op) {
        ++fp.pos;
        push();
        skipTo(">\n");
        if (*fp.pos == '\n') {
          // TODO: Print proper error
          // "Unable to find closing >"
          return NULL;
        }
        token_t *token = new headerToken(getFileOrigin(),
                                         true,
                                         str());
        ++fp.pos;
        pop();
        return token;
      }
      if (!(type & tokenType::string)) {
        // TODO: Print proper error
        // "Not able to parse header"
        return NULL;
      }
      std::string value;
      getString(value);
      return new headerToken(getFileOrigin(), false, value);
    }
    //==================================

    //---[ Tokenizer ]------------------

    //==================================
  }
}
