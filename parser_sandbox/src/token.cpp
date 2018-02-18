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

namespace occa {
  namespace lang {
    namespace charcodes {
      const char whitespace[]          = " \n\t\r\v\f";
      const char whitespaceNoNewline[] = " \t\r\v\f";
      const char alpha[]               = ("abcdefghijklmnopqrstuvwxyz"
                                          "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
      const char number[]              = "0123456789";
      const char alphanumber[]         = ("abcdefghijklmnopqrstuvwxyz"
                                          "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                          "0123456789");
      const char identifierStart[]     = ("abcdefghijklmnopqrstuvwxyz"
                                          "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                          "_");
      const char identifier[]          = ("abcdefghijklmnopqrstuvwxyz"
                                          "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                          "0123456789"
                                          "_");
      const char operators[]           = "!#%&()*+,-./:;<=>?[]^{|}~";
    }

    void getOperators(operatorTrie &operators) {
      operators.add(op::not_.str             , &op::not_);
      operators.add(op::tilde.str            , &op::tilde);
      operators.add(op::leftIncrement.str    , &op::leftIncrement);
      operators.add(op::leftDecrement.str    , &op::leftDecrement);

      operators.add(op::add.str              , &op::add);
      operators.add(op::sub.str              , &op::sub);
      operators.add(op::mult.str             , &op::mult);
      operators.add(op::div.str              , &op::div);
      operators.add(op::mod.str              , &op::mod);

      operators.add(op::lessThan.str         , &op::lessThan);
      operators.add(op::lessThanEq.str       , &op::lessThanEq);
      operators.add(op::equal.str            , &op::equal);
      operators.add(op::notEqual.str         , &op::notEqual);
      operators.add(op::greaterThan.str      , &op::greaterThan);
      operators.add(op::greaterThanEq.str    , &op::greaterThanEq);

      operators.add(op::and_.str             , &op::and_);
      operators.add(op::or_.str              , &op::or_);

      operators.add(op::bitAnd.str           , &op::bitAnd);
      operators.add(op::bitOr.str            , &op::bitOr);
      operators.add(op::xor_.str             , &op::xor_);
      operators.add(op::leftShift.str        , &op::leftShift);
      operators.add(op::rightShift.str       , &op::rightShift);

      operators.add(op::assign.str           , &op::assign);
      operators.add(op::addEq.str            , &op::addEq);
      operators.add(op::subEq.str            , &op::subEq);
      operators.add(op::multEq.str           , &op::multEq);
      operators.add(op::divEq.str            , &op::divEq);
      operators.add(op::modEq.str            , &op::modEq);
      operators.add(op::andEq.str            , &op::andEq);
      operators.add(op::orEq.str             , &op::orEq);
      operators.add(op::xorEq.str            , &op::xorEq);
      operators.add(op::leftShiftEq.str      , &op::leftShiftEq);
      operators.add(op::rightShiftEq.str     , &op::rightShiftEq);

      operators.add(op::comma.str            , &op::comma);
      operators.add(op::scope.str            , &op::scope);
      operators.add(op::dot.str              , &op::dot);
      operators.add(op::dotStar.str          , &op::dotStar);
      operators.add(op::arrow.str            , &op::arrow);
      operators.add(op::arrowStar.str        , &op::arrowStar);
      operators.add(op::ternary.str          , &op::ternary);
      operators.add(op::colon.str            , &op::colon);

      operators.add(op::braceStart.str       , &op::braceStart);
      operators.add(op::braceEnd.str         , &op::braceEnd);
      operators.add(op::bracketStart.str     , &op::bracketStart);
      operators.add(op::bracketEnd.str       , &op::bracketEnd);
      operators.add(op::parenthesesStart.str , &op::parenthesesStart);
      operators.add(op::parenthesesEnd.str   , &op::parenthesesEnd);

      operators.add(op::lineComment.str      , &op::lineComment);
      operators.add(op::blockCommentStart.str, &op::blockCommentStart);
      operators.add(op::blockCommentEnd.str  , &op::blockCommentEnd);

      operators.add(op::hash.str             , &op::hash);
      operators.add(op::hashhash.str         , &op::hashhash);

      operators.add(op::semicolon.str        , &op::semicolon);
      operators.add(op::ellipsis.str         , &op::ellipsis);

      operators.add(op::cudaCallStart.str    , &op::cudaCallStart);
      operators.add(op::cudaCallEnd.str      , &op::cudaCallEnd);
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
      const int none          = 0;
      const int unknown       = (1 << 0);

      const int newline       = (1 << 1);

      const int identifier    = (1 << 2);

      const int systemHeader  = (1 << 3);
      const int header        = (3 << 3);

      const int primitive     = (1 << 5);
      const int op            = (1 << 6);

      const int char_         = (1 << 7);
      const int string        = (1 << 8);
      const int withUDF       = (1 << 9);
      const int withEncoding  = ((encodingType::ux |
                                  encodingType::R) << 10);
      const int encodingShift = 10;

      int getEncoding(const int type) {
        return ((type & withEncoding) >> encodingShift);
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

    void token_t::preprint(std::ostream &out) {
      origin.preprint(out);
    }

    void token_t::postprint(std::ostream &out) {
      origin.postprint(out);
    }

    std::string token_t::str() const {
      std::stringstream ss;
      print(ss);
      return ss.str();
    }

    //---[ Unknown ]--------------------
    unknownToken::unknownToken(const fileOrigin &origin_) :
      token_t(origin_) {}

    unknownToken::~unknownToken() {}

    int unknownToken::type() const {
      return tokenType::unknown;
    }

    token_t* unknownToken::clone() {
      return new unknownToken(origin);
    }

    void unknownToken::print(std::ostream &out) const {
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

    token_t* newlineToken::clone() {
      return new newlineToken(origin);
    }

    void newlineToken::print(std::ostream &out) const {
      out << '\n';
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

    token_t* identifierToken::clone() {
      return new identifierToken(origin, value);
    }

    void identifierToken::print(std::ostream &out) const {
      out << value;
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

    token_t* primitiveToken::clone() {
      return new primitiveToken(origin, value, strValue);
    }

    void primitiveToken::print(std::ostream &out) const {
      out << strValue;
    }
    //==================================

    //---[ Operator ]-------------------
    operatorToken::operatorToken(const fileOrigin &origin_,
                                 const operator_t &op_) :
      token_t(origin_),
      op(op_) {}

    operatorToken::~operatorToken() {}

    int operatorToken::type() const {
      return tokenType::op;
    }

    token_t* operatorToken::clone() {
      return new operatorToken(origin, op);
    }

    void operatorToken::print(std::ostream &out) const {
      out << op.str;
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

    token_t* charToken::clone() {
      return new charToken(origin, encoding, value, udf);
    }

    void charToken::print(std::ostream &out) const {
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

    token_t* stringToken::clone() {
      return new stringToken(origin, encoding, value, udf);
    }

    void stringToken::append(const stringToken &token) {
      origin.position.end = token.origin.position.end;

      encoding = tokenType::mergeEncodings(encoding,
                                           token.encoding);
      value += token.value;
      udf = token.udf;
    }

    void stringToken::print(std::ostream &out) const {
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

    //---[ Header ]---------------------
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

    token_t* headerToken::clone() {
      return new headerToken(origin, systemHeader, value);
    }

    void headerToken::print(std::ostream &out) const {
      if (systemHeader) {
        out << '<' << value << '>';
      } else {
        out << '"' << value << '"';
      }
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
    //==================================
  }
}
