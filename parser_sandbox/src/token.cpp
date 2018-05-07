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
#include "type.hpp"
#include "variable.hpp"

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

      const int pragma        = (1 << 5);

      const int identifier    = (1 << 6);

      const int qualifier     = (1 << 7);
      const int type          = (1 << 8);
      const int variable      = (1 << 9);

      const int primitive     = (1 << 10);
      const int op            = (1 << 11);

      const int char_         = (1 << 12);
      const int string        = (1 << 13);
      const int withUDF       = (1 << 14);
      const int withEncoding  = ((encodingType::ux |
                                  encodingType::R) << 15);
      const int encodingShift = 15;

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

    opType_t token_t::getOpType() {
      if (type() != tokenType::op) {
        return operatorType::none;
      }
      return to<operatorToken>().opType();
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

    void token_t::debugPrint() const {
      std::cerr << '[';
      print(std::cerr);
      std::cerr << "]\n";
    }

    std::ostream& operator << (std::ostream &out, token_t &token) {
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

    //---[ Pragma ]---------------------
    pragmaToken::pragmaToken(const fileOrigin &origin_,
                             const std::string &value_) :
      token_t(origin_),
      value(value_) {}

    pragmaToken::~pragmaToken() {}

    int pragmaToken::type() const {
      return tokenType::pragma;
    }

    token_t* pragmaToken::clone() {
      return new pragmaToken(origin, value);
    }

    void pragmaToken::print(std::ostream &out) const {
      out << "#pragma " << value << '\n';
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

    //---[ Qualifier ]------------------
    qualifierToken::qualifierToken(const fileOrigin &origin_,
                                   const qualifier_t &qualifier_) :
      token_t(origin_),
      qualifier(qualifier_) {}

    qualifierToken::~qualifierToken() {}

    int qualifierToken::type() const {
      return tokenType::qualifier;
    }

    token_t* qualifierToken::clone() {
      return new qualifierToken(origin, qualifier);
    }

    void qualifierToken::print(std::ostream &out) const {
      out << qualifier.name;
    }
    //==================================

    //---[ Type ]-----------------------
    typeToken::typeToken(const fileOrigin &origin_,
                         const type_t &type__) :
    token_t(origin_),
    type_(type__) {}

    typeToken::~typeToken() {}

    int typeToken::type() const {
      return tokenType::type;
    }

    token_t* typeToken::clone() {
      return new typeToken(origin, type_);
    }

    void typeToken::print(std::ostream &out) const {
      out << type_.name();
    }
    //==================================

    //---[ Variable ]-------------------
    variableToken::variableToken(const fileOrigin &origin_,
                                 const variable_t &var_) :
      token_t(origin_),
      var(var_) {}

    variableToken::~variableToken() {}

    int variableToken::type() const {
      return tokenType::variable;
    }

    token_t* variableToken::clone() {
      return new variableToken(origin, var);
    }

    void variableToken::print(std::ostream &out) const {
      out << var.name();
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
      op(&op_) {}

    operatorToken::~operatorToken() {}

    int operatorToken::type() const {
      return tokenType::op;
    }

    const opType_t& operatorToken::opType() const {
      return op->opType;
    }

    token_t* operatorToken::clone() {
      return new operatorToken(origin, *op);
    }

    void operatorToken::print(std::ostream &out) const {
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
      const int tokenCount = (int) tokens.size();
      for (int i = 0; i < tokenCount; ++i) {
        tokens[i]->print(ss);
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
