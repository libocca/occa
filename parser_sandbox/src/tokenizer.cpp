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
    namespace encodingType {
      const int none = 0;
      const int R    = (1 << 0);
      const int u8   = (1 << 1);
      const int u    = (1 << 2);
      const int U    = (1 << 3);
      const int L    = (1 << 4);
      const int ux   = (u8 | u | U | L);
    }

    namespace tokenType {
      const int none         = 0;

      const int identifier   = (1 << 0);

      const int systemHeader = (1 << 1);
      const int header       = (3 << 1);

      const int primitive    = (1 << 3);
      const int op           = (1 << 4);

      const int withUType    = (1 << 5);
      const int withUDF      = (1 << 6);
      const int char_        = (1 << 7);
      const int string       = (1 << 8);
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
      if (uType & encodingType::u) {
        pout << 'u';
      } else if (uType & encodingType::U) {
        pout << 'U';
      } else if (uType & encodingType::L) {
        pout << 'L';
      }
      pout << '\'' << value << '\'' << udf;
    }

    stringToken::~stringToken() {}

    stringToken::stringToken(const fileOrigin &origin_,
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
    charStream::charStream(const char *root) :
      file(NULL),
      fp(root) {}

    charStream::charStream(file_t *file_,
                           const char *root) :
      file(file_),
      fp(root) {}

    charStream::charStream(const charStream &stream) :
      file(stream.file),
      fp(stream.fp),
      stack(stream.stack) {}

    const char *charStream::getPosition() {
      return fp.pos;
    }

    void charStream::setPosition(const char * pos) {
      fp.pos = pos;
    }

    int charStream::getLine() {
      return fp.line;
    }

    void charStream::setLine(const int line) {
      fp.line = line;
    }

    fileOrigin charStream::getFileOrigin() {
      return fileOrigin(file, fp);
    }

    void charStream::push() {
      stack.push_back(
        filePosition(fp.line,
                     fp.lineStart,
                     fp.pos)
      );
    }

    void charStream::pop(const bool rewind) {
      if (stack.size() > 0) {
        if (rewind) {
          fp = stack.back();
        }
        stack.pop_back();
      }
    }

    void charStream::popAndRewind() {
      pop(true);
    }

    std::string charStream::str() {
      if (stack.size() == 0) {
        return "";
      }
      filePosition last = stack.back();
      return std::string(last.pos, fp.pos - last.pos);
    }

    void charStream::countSkippedLines(const char *start) {
      const char *pos = start;
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

    void charStream::skipTo(const char delimiter) {
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

    void charStream::skipTo(const char *delimiters) {
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

    void charStream::skipFrom(const char *delimiters) {
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

    int charStream::peekForIdentifier() {
      push();
      ++fp.pos;
      skipFrom(charcodes::identifier);
      const std::string identifier = str();
      int type = shallowPeek();
      popAndRewind();

      // [u8R]"foo" or [u]'foo'
      if ((type & tokenType::string) &&
          getStringEncoding(identifier)) {
        // u8R["foo"]
        return (tokenType::withUType
                | tokenType::string);
      }
      if ((type & tokenType::char_) &&
          getCharacterEncoding(identifier)) {
        // u['foo']
        return (tokenType::withUType
                | tokenType::char_);
      }
      return tokenType::identifier;
    }

    int charStream::peekForHeader() {
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

    token_t* charStream::getToken() {
      skipWhitespace();
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
                 lex::charIsIn(*fp.pos, charcodes::identifierStart));
      push();
      ++fp.pos;
      skipFrom(charcodes::identifier);
      return new identifierToken(getFileOrigin(),
                                 str());
      pop();
    }

    token_t* charStream::getPrimitiveToken() {
      const char *start = fp.pos;
      primitive value = primitive::load(fp.pos);
      // TODO: Print proper error
      OCCA_ERROR("Not able to parse primitive",
                 !value.isNaN());
      countSkippedLines(start);
      return new primitiveToken(getFileOrigin(),
                                value);
    }

    token_t* charStream::getOperatorToken() {
      operatorTrie &operators = getOperators();
      operatorTrie::result_t result = operators.getLongest(fp.pos);
      // TODO: Print proper error
      OCCA_ERROR("Not able to parse operator",
                 result.success());
      return new operatorToken(getFileOrigin(),
                               *(result.value()));
    }

    token_t* charStream::getStringToken() {
      if (*fp.pos != '"') {
        // TODO: Print proper error
        OCCA_FORCE_ERROR("Not able to parse string");
      }
      ++fp.pos;
      push();
      skipTo("\"\n");
      if (*fp.pos == '\n') {
        // TODO: Print proper error
        OCCA_FORCE_ERROR("Unable to find closing \"");
      }
      token_t *token = new stringToken(getFileOrigin(),
                                       encodingType::u8,
                                       str(),
                                       "_km");
      pop();
      ++fp.pos;
      return token;
    }

    token_t* charStream::getCharToken() {
      if (*fp.pos != '\'') {
        // TODO: Print proper error
        OCCA_FORCE_ERROR("Not able to parse string");
      }
      ++fp.pos;
      push();
      skipTo("\'\n");
      if (*fp.pos == '\n') {
        // TODO: Print proper error
        OCCA_FORCE_ERROR("Unable to find closing '");
      }
      token_t *token = new charToken(getFileOrigin(),
                                     encodingType::u,
                                     str(),
                                     "_km");
      pop();
      ++fp.pos;
      return token;
    }

    token_t* charStream::getHeaderToken() {
      int type = shallowPeek();
      if (type & tokenType::op) {
        ++fp.pos;
        push();
        token_t *token = new headerToken(getFileOrigin(),
                                         true,
                                         str());
        skipTo(">\n");
        if (*fp.pos == '\n') {
          // TODO: Print proper error
          OCCA_FORCE_ERROR("Unable to find closing >");
        }
        ++fp.pos;
        pop();
        return token;
      }
      if (type & tokenType::string) {
        stringToken &token = *((stringToken*) getStringToken());
        std::string value = token.value;
        delete &token;
        return new headerToken(getFileOrigin(), false, value);
      }
      // TODO: Print proper error
      OCCA_FORCE_ERROR("Not able to parse header");
      return new headerToken(getFileOrigin(), false, "");
    }
    //==================================

    //---[ Tokenizer ]------------------

    //==================================
  }
}
