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
#include "occa/tools/string.hpp"
#include "token.hpp"

namespace occa {
  namespace lang {
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

    tokenizer::tokenizer(const char *root) :
      lastToken(NULL),
      origin(filePosition(root)),
      fp(origin.position) {
      getOperators(operators);
    }

    tokenizer::tokenizer(file_t *file_,
                         const char *root) :
      lastToken(NULL),
      origin(*file_, filePosition(root)),
      fp(origin.position) {
      getOperators(operators);
    }

    tokenizer::tokenizer(fileOrigin origin_) :
      lastToken(NULL),
      origin(origin_),
      fp(origin.position) {
      getOperators(operators);
    }

    tokenizer::tokenizer(const tokenizer &stream) :
      lastToken(stream.lastToken),
      origin(stream.origin),
      fp(origin.position),
      stack(stream.stack),
      sourceStack(stream.sourceStack) {
      getOperators(operators);
    }


    tokenizer& tokenizer::operator = (const tokenizer &stream) {
      lastToken   = stream.lastToken;
      origin      = stream.origin;
      stack       = stream.stack;
      sourceStack = stream.sourceStack;
      return *this;
    }

    tokenizer::~tokenizer() {}

    void tokenizer::preprint(std::ostream &out) {
      origin.preprint(out);
    }

    void tokenizer::postprint(std::ostream &out) {
      origin.postprint(out);
    }

    void tokenizer::setLine(const int line) {
      fp.line = line;
    }

    baseStream<token_t*>& tokenizer::clone() const {
      return *(new tokenizer(*this));
    }

    bool tokenizer::reachedTheEnd() const {
      return ((*fp.start == '\0') &&
              !stack.size());
    }

    bool tokenizer::isEmpty() {
      while (!reachedTheEnd() &&
             !lastToken) {
        lastToken = getToken();
      }
      return !lastToken;
    }

    void tokenizer::setNext(token_t *&out) {
      if (!isEmpty()) {
        out = lastToken;
        lastToken = NULL;
      } else {
        out = NULL;
      }
    }

    void tokenizer::pushSource(const bool fromInclude,
                               file_t *file,
                               const filePosition &position) {
      sourceStack.push_back(stack);
      origin.push(fromInclude,
                  *file,
                  position);
    }

    void tokenizer::popSource() {
      OCCA_ERROR("Unable to call tokenizer::popSource()",
                 sourceStack.size() > 0);
      stack = sourceStack.back();
      sourceStack.pop_back();
    }

    void tokenizer::push() {
      stack.push_back(origin);
    }

    void tokenizer::pop(const bool rewind) {
      OCCA_ERROR("Unable to call tokenizer::pop()",
                 stack.size() > 0);
      if (rewind) {
        origin = stack.back();
      }
      stack.pop_back();
    }

    void tokenizer::popAndRewind() {
      pop(true);
    }

    fileOrigin tokenizer::popTokenOrigin() {
      const size_t size = strSize();
      fileOrigin tokenOrigin = stack.back();
      tokenOrigin.position.end = tokenOrigin.position.start + size;
      pop();
      return tokenOrigin;
    }

    size_t tokenizer::strSize() {
      if (stack.size() == 0) {
        printError("Not able to strSize() without a stack");
        return 0;
      }
      fileOrigin last = stack.back();
      return (fp.start - last.position.start);
    }

    std::string tokenizer::str() {
      if (stack.size() == 0) {
        printError("Not able to str() without a stack");
        return "";
      }
      fileOrigin last = stack.back();
      return std::string(last.position.start,
                         fp.start - last.position.start);
    }

    void tokenizer::countSkippedLines() {
      if (stack.size() == 0) {
        printError("Not able to countSkippedLines() without a stack");
        return;
      }
      fileOrigin last = stack.back();
      if (last.file != origin.file) {
        printError("Trying to countSkippedLines() across different files");
        return;
      }
      const char *pos = last.position.start;
      const char *end = fp.start;
      while (pos < end) {
        if (*pos == '\\') {
          if (fp.start[1] == '\n') {
            fp.lineStart = fp.start + 2;
            ++fp.line;
          }
          pos += 1 + (pos[1] != '\0');
          continue;
        }
        if (*pos == '\n') {
          fp.lineStart = fp.start + 1;
          ++fp.line;
        }
        ++pos;
      }
    }

    void tokenizer::skipTo(const char delimiter) {
      while (*fp.start != '\0') {
        if (*fp.start == '\\') {
          if (fp.start[1] == '\n') {
            fp.lineStart = fp.start + 2;
            ++fp.line;
          }
          fp.start += 1 + (fp.start[1] != '\0');
          continue;
        }
        if (*fp.start == delimiter) {
          return;
        }
        if (*fp.start == '\n') {
          fp.lineStart = fp.start + 1;
          ++fp.line;
        }
        ++fp.start;
      }
    }

    void tokenizer::skipTo(const char *delimiters) {
      while (*fp.start != '\0') {
        if (*fp.start == '\\') {
          if (fp.start[1] == '\n') {
            fp.lineStart = fp.start + 2;
            ++fp.line;
          }
          fp.start += 1 + (fp.start[1] != '\0');
          continue;
        }
        if (lex::charIsIn(*fp.start, delimiters)) {
          return;
        }
        if (*fp.start == '\n') {
          fp.lineStart = fp.start + 1;
          ++fp.line;
        }
        ++fp.start;
      }
    }

    void tokenizer::skipFrom(const char *delimiters) {
      while (*fp.start != '\0') {
        if (*fp.start == '\\') {
          if (fp.start[1] == '\n') {
            fp.lineStart = fp.start + 2;
            ++fp.line;
          }
          fp.start += 1 + (fp.start[1] != '\0');
          continue;
        }
        if (!lex::charIsIn(*fp.start, delimiters)) {
          break;
        }
        if (*fp.start == '\n') {
          fp.lineStart = fp.start + 1;
          ++fp.line;
        }
        ++fp.start;
      }
    }

    void tokenizer::skipWhitespace() {
      skipFrom(charcodes::whitespaceNoNewline);
    }

    int tokenizer::peek() {
      int type = shallowPeek();
      if (type & tokenType::identifier) {
        type = peekForIdentifier();
      } else if (type & tokenType::op) {
        type = peekForOperator();
      }
      return type;
    }

    int tokenizer::shallowPeek() {
      skipWhitespace();

      const char c = *fp.start;
      if (c == '\0') {
        return tokenType::none;
      }
      // Primitive must be checked before identifiers
      //   and operators since:
      //   - true/false
      //   - Operators can start with a . (for example, .01)
      const char *pos = fp.start;
      if (primitive::load(pos, false).type != occa::primitiveType::none) {
        return tokenType::primitive;
      }
      if (lex::charIsIn(c, charcodes::identifierStart)) {
        return tokenType::identifier;
      }
      if (lex::charIsIn(c, charcodes::operators)) {
        return tokenType::op;
      }
      if (c == '\n') {
        return tokenType::newline;
      }
      if (c == '"') {
        return tokenType::string;
      }
      if (c == '\'') {
        return tokenType::char_;
      }
      return tokenType::none;
    }

    int tokenizer::peekForIdentifier() {
      push();
      ++fp.start;
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
        const int encoding = getCharacterEncoding(identifier);
        if (encoding) {
          return (tokenType::char_ |
                  (encoding << tokenType::encodingShift));
        }
      }
      return tokenType::identifier;
    }

    int tokenizer::peekForOperator() {
      push();
      operatorTrie::result_t result = operators.getLongest(fp.start);
      if (!result.success()) {
        printError("Not able to parse operator");
        popAndRewind();
        return tokenType::none;
      }
      const operator_t &op = *(result.value());
      if (op.opType & operatorType::comment) {
        pop();
        if (op.opType == operatorType::lineComment) {
          return skipLineCommentAndPeek();
        }
        else if (op.opType == operatorType::blockCommentStart) {
          return skipBlockCommentAndPeek();
        }
        else {
          printError("Couldn't find an opening /*");
          return peek();
        }
      }
      popAndRewind();
      return tokenType::op;
    }

    int tokenizer::peekForHeader() {
      int type = shallowPeek();
      if (type & tokenType::op) {
        push();
        operatorTrie::result_t result = operators.getLongest(fp.start);
        popAndRewind();
        if (result.success() &&
            (result.value()->opType & operatorType::lessThan)) {
          return tokenType::systemHeader;
        }
      }
      else if (type & tokenType::string) {
        return tokenType::header;
      }
      return tokenType::none;
    }

    void tokenizer::getIdentifier(std::string &value) {
      if (!lex::charIsIn(*fp.start, charcodes::identifierStart)) {
        return;
      }
      push();
      ++fp.start;
      skipFrom(charcodes::identifier);
      value = str();
      pop();
    }

    void tokenizer::getString(std::string &value,
                              const int encoding) {
      if (encoding & encodingType::R) {
        getRawString(value);
        return;
      }
      if (*fp.start != '"') {
        return;
      }
      push();
      ++fp.start;
      push();
      skipTo("\"\n");
      if (*fp.start == '\n') {
        pop();
        popAndRewind();
        return;
      }
      value = unescape(str(), '"');
      pop();
      pop();
      ++fp.start;
    }

    void tokenizer::getRawString(std::string &value) {
      // TODO: Keep the delimiter(s)
      if (*fp.start != '"') {
        return;
      }
      push();
      ++fp.start; // Skip "
      push();

      // Find delimiter
      skipTo("(\n");
      if (*fp.start == '\n') {
        pop();
        popAndRewind();
        return;
      }

      // Find end pattern
      std::string end;
      end += ')';
      end += str();
      end += '"';
      pop();
      ++fp.start; // Skip (
      push();

      // Find end match
      const int chars = (int) end.size();
      const char *m   = end.c_str();
      int mi;
      while (*fp.start != '\0') {
        for (mi = 0; mi < chars; ++mi) {
          if (fp.start[mi] != m[mi]) {
            break;
          }
        }
        if (mi == chars) {
          break;
        }
        if (*fp.start == '\n') {
          fp.lineStart = fp.start + 1;
          ++fp.line;
        }
        ++fp.start;
      }

      // Make sure we found delimiter
      if (*fp.start == '\0') {
        pop();
        popAndRewind();
        return;
      }
      value = str();
      fp.start += chars;
    }

    int tokenizer::skipLineCommentAndPeek() {
      skipTo('\n');
      return (fp.start
              ? tokenType::newline
              : tokenType::none);
    }

    int tokenizer::skipBlockCommentAndPeek() {
      while (*fp.start != '\0') {
        skipTo('*');
        if (*fp.start == '*') {
          ++fp.start;
          if (*fp.start == '/') {
            ++fp.start;
            skipWhitespace();
            return peek();
          }
        }
      }
      return tokenType::none;
    }

    token_t* tokenizer::getToken() {
      if (reachedTheEnd()) {
        return NULL;
      }

      skipWhitespace();

      // Check if file finished
      bool finishedSource = false;
      while ((*fp.start == '\0') && stack.size()) {
        popSource();
        skipWhitespace();
        finishedSource = true;
      }
      if (finishedSource) {
        push();
        ++fp.start;
        return new newlineToken(popTokenOrigin());
      }

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
      if (type & tokenType::newline) {
        push();
        ++fp.start;
        ++fp.line;
        fp.lineStart = fp.start;
        return new newlineToken(popTokenOrigin());
      }
      if (type & tokenType::char_) {
        return getCharToken(tokenType::getEncoding(type));
      }
      if (type & tokenType::string) {
        return getStringToken(tokenType::getEncoding(type));
      }

      push();
      ++fp.start;
      return new unknownToken(popTokenOrigin());
    }

    token_t* tokenizer::getIdentifierToken() {
      if (!lex::charIsIn(*fp.start, charcodes::identifierStart)) {
        printError("Not able to parse identifier");
        return NULL;
      }

      push();
      std::string value;
      getIdentifier(value);

      return new identifierToken(popTokenOrigin(),
                                 value);
    }

    token_t* tokenizer::getPrimitiveToken() {
      push();
      primitive value = primitive::load(fp.start);
      if (value.isNaN()) {
        printError("Not able to parse primitive");
        popAndRewind();
        return NULL;
      }
      const std::string strValue = str();
      countSkippedLines();
      return new primitiveToken(popTokenOrigin(),
                                value,
                                strValue);
    }

    token_t* tokenizer::getOperatorToken() {
      push();
      operatorTrie::result_t result = operators.getLongest(fp.start);
      if (!result.success()) {
        printError("Not able to parse operator");
        return NULL;
      }
      fp.start += result.length; // Skip operator
      return new operatorToken(popTokenOrigin(),
                               *(result.value()));
    }

    token_t* tokenizer::getStringToken(const int encoding) {
      push();

      if (encoding) {
        std::string encodingStr;
        getIdentifier(encodingStr);
      }

      if (*fp.start != '"') {
        printError("Not able to parse string");
        pop();
        return NULL;
      }

      const char *start = fp.start;
      std::string value, udf;
      getString(value, encoding);
      if (fp.start == start) {
        printError("Not able to find closing \"");
        pop();
        return NULL;
      }

      if (*fp.start == '_') {
        getIdentifier(udf);
      }

      return new stringToken(popTokenOrigin(),
                             encoding, value, udf);
    }

    token_t* tokenizer::getCharToken(const int encoding) {
      push();

      if (encoding) {
        std::string encodingStr;
        getIdentifier(encodingStr);
      }
      if (*fp.start != '\'') {
        printError("Not able to parse char");
        pop();
        return NULL;
      }

      ++fp.start; // Skip '
      push();
      skipTo("'\n");
      if (*fp.start == '\n') {
        printError("Not able to find closing '");
        popAndRewind();
        pop();
        return NULL;
      }
      const std::string value = unescape(str(), '\'');
      ++fp.start;
      pop();

      std::string udf;
      if (*fp.start == '_') {
        getIdentifier(udf);
      }

      return new charToken(popTokenOrigin(),
                           encoding, value, udf);
    }

    token_t* tokenizer::getHeaderToken() {
      int type = shallowPeek();

      // Push after in case of whitespace
      push();
      if (type & tokenType::op) {
        ++fp.start; // Skip <
        push();
        skipTo(">\n");
        if (*fp.start == '\n') {
          printError("Not able to find closing >");
          pop();
          pop();
          return NULL;
        }
        const std::string header = str();
        pop();
        ++fp.start; // Skip >
        return new headerToken(popTokenOrigin(),
                               true, header);
      }

      if (!(type & tokenType::string)) {
        printError("Not able to parse header");
        return NULL;
      }

      std::string value;
      getString(value);

      return new headerToken(popTokenOrigin(),
                             false, value);
    }

    tokenVector tokenizer::tokenize(const std::string &source) {
      tokenVector tokens;
      fileOrigin origin = originSource::string;
      tokenize(tokens, origin, source);
      return tokens;
    }

    void tokenizer::tokenize(tokenVector &tokens,
                             fileOrigin origin,
                             const std::string &source) {
      fileOrigin fakeOrigin(*origin.file,
                            source.c_str());

      tokenizer tstream(fakeOrigin);
      // Fill tokens
      token_t *token;
      while (!tstream.isEmpty()) {
        tstream.setNext(token);
        tokens.push_back(token);
      }
    }
  }
}
