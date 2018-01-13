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
#include "occa/tools/testing.hpp"

#include "tokenizer.hpp"

void testSkipMethods();
void testPushPop();
void testPeekMethods();
void testTokenMethods();

std::string streamSource;
occa::lang::charStream stream(NULL);
occa::lang::token_t *token = NULL;

int main(const int argc, const char **argv) {
  testSkipMethods();
  testPushPop();
  testPeekMethods();
  testTokenMethods();
  if (token) {
    delete token;
  }
  return 0;
}

void setStream(const std::string &s) {
  streamSource = s;
  stream = occa::lang::charStream(NULL, streamSource.c_str());
}

void setToken(const std::string &s) {
  setStream(s);
  if (token) {
    delete token;
  }
  token = stream.getToken();
}

void setHeaderToken(const std::string &s) {
  setStream(s);
  if (token) {
    delete token;
  }
  token = stream.getHeaderToken();
}

int tokenType() {
  return token ? token->type() : 0;
}

void testSkipMethods() {
  setStream("a\nb\nc\n\n\n\n\n\nd\ne");
  const char *c = streamSource.c_str();

  stream.skipTo('b');
  OCCA_ASSERT_EQUAL('b', *stream.getPosition());
  stream.skipTo('e');
  OCCA_ASSERT_EQUAL('e', *stream.getPosition());

  stream.setPosition(c);
  stream.skipTo("c\n");
  OCCA_ASSERT_EQUAL(c + 1, stream.getPosition());

  stream.setPosition(c + 5);
  stream.skipFrom("\n");
  OCCA_ASSERT_EQUAL('d', *stream.getPosition());
}

void testPushPop() {
  setStream("a\nb\nc\nd\ne");
  const char *c = streamSource.c_str();

  stream.push();
  stream.skipTo('c');
  OCCA_ASSERT_EQUAL(3,
                    stream.getLine());
  OCCA_ASSERT_EQUAL(c + 4,
                    stream.getPosition());
  stream.popAndRewind();
  OCCA_ASSERT_EQUAL(1,
                    stream.getLine());
  OCCA_ASSERT_EQUAL(c + 0,
                    stream.getPosition());
  stream.push();
  stream.push();
  stream.push();
  stream.skipTo('c');
  stream.pop();
  stream.pop();
  stream.pop();
  OCCA_ASSERT_EQUAL(3,
                    stream.getLine());
  OCCA_ASSERT_EQUAL(c + 4,
                    stream.getPosition());
}

#define testStringPeek(s, encoding)                                     \
  setStream(s);                                                         \
  OCCA_ASSERT_EQUAL_BINARY((encoding << occa::lang::tokenType::encodingShift) | \
                           occa::lang::tokenType::string,               \
                           stream.peek())

#define testCharPeek(s, encoding)                                       \
  setStream(s);                                                         \
  OCCA_ASSERT_EQUAL_BINARY((encoding << occa::lang::tokenType::encodingShift) | \
                           occa::lang::tokenType::char_,                \
                           stream.peek())

void testPeekMethods() {
  setStream("abcd");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::identifier,
                           stream.peek());
  setStream("_abcd");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::identifier,
                           stream.peek());
  setStream("_abcd020230");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::identifier,
                           stream.peek());

  setStream("1");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::primitive,
                           stream.peek());
  setStream("+2.0");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::primitive,
                           stream.peek());
  setStream("+2.0e10");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::primitive,
                           stream.peek());
  setStream("+.0e10-10");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::primitive,
                           stream.peek());

  setStream("+");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::op,
                           stream.peek());
  setStream("<");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::op,
                           stream.peek());
  setStream("->*");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::op,
                           stream.peek());
  setStream("==");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::op,
                           stream.peek());

  testStringPeek("\"\""                , 0);
  testStringPeek("\"string\\\"string\"", 0);
  testStringPeek("R\"string\""         , occa::lang::encodingType::R);
  testStringPeek("u8\"string\""        , occa::lang::encodingType::u8);
  testStringPeek("u\"string\""         , occa::lang::encodingType::u);
  testStringPeek("U\"string\""         , occa::lang::encodingType::U);
  testStringPeek("L\"string\""         , occa::lang::encodingType::L);
  testStringPeek("u8R\"string\""       , (occa::lang::encodingType::u8 |
                                          occa::lang::encodingType::R));
  testStringPeek("uR\"string\""        , (occa::lang::encodingType::u |
                                          occa::lang::encodingType::R));
  testStringPeek("UR\"string\""        , (occa::lang::encodingType::U |
                                          occa::lang::encodingType::R));
  testStringPeek("LR\"string\""        , (occa::lang::encodingType::L |
                                          occa::lang::encodingType::R));

  testCharPeek("'a'"    , 0);
  testCharPeek("'\\''"  , 0);
  testCharPeek("u'\\''" , occa::lang::encodingType::u);
  testCharPeek("U'\\''" , occa::lang::encodingType::U);
  testCharPeek("L'\\''" , occa::lang::encodingType::L);

  setStream("<foobar>");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::systemHeader,
                           stream.peekForHeader());
  setStream("\"foobar\"");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::header,
                           stream.peekForHeader());
}

#define testStringToken(s, encoding_)                                   \
  setToken(s);                                                          \
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::string,               \
                           tokenType())                                 \
  OCCA_ASSERT_EQUAL_BINARY(encoding_,                                   \
                           token->to<occa::lang::stringToken>().encoding)

#define testCharToken(s, encoding_)                                     \
  setToken(s);                                                          \
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::char_,                \
                           tokenType())                                 \
  OCCA_ASSERT_EQUAL_BINARY(encoding_,                                   \
                           token->to<occa::lang::charToken>().encoding)

void testTokenMethods() {
  setToken("abcd");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::identifier,
                           tokenType());
  setToken("_abcd");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::identifier,
                           tokenType());
  setToken("_abcd020230");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::identifier,
                           tokenType());

  setToken("1");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::primitive,
                           tokenType());
  setToken("+2.0");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::primitive,
                           tokenType());
  setToken("+2.0e10");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::primitive,
                           tokenType());
  setToken("+.0e10-10");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::primitive,
                           tokenType());

  setToken("+");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::op,
                           tokenType());
  setToken("<");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::op,
                           tokenType());
  setToken("->*");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::op,
                           tokenType());
  setToken("==");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::op,
                           tokenType());

  testStringToken("\"\""                , 0);
  testStringToken("\"string\\\"string\"", 0);
  testStringToken("R\"string\""         , occa::lang::encodingType::R);
  testStringToken("u8\"string\""        , occa::lang::encodingType::u8);
  testStringToken("u\"string\""         , occa::lang::encodingType::u);
  testStringToken("U\"string\""         , occa::lang::encodingType::U);
  testStringToken("L\"string\""         , occa::lang::encodingType::L);
  testStringToken("u8R\"string\""       , (occa::lang::encodingType::u8 |
                                           occa::lang::encodingType::R));
  testStringToken("uR\"string\""        , (occa::lang::encodingType::u |
                                           occa::lang::encodingType::R));
  testStringToken("UR\"string\""        , (occa::lang::encodingType::U |
                                           occa::lang::encodingType::R));
  testStringToken("LR\"string\""        , (occa::lang::encodingType::L |
                                           occa::lang::encodingType::R));

  testCharToken("'a'"    , 0);
  testCharToken("'\\''"  , 0);
  testCharToken("u'\\''" , occa::lang::encodingType::u);
  testCharToken("U'\\''" , occa::lang::encodingType::U);
  testCharToken("L'\\''" , occa::lang::encodingType::L);

  setHeaderToken("<foobar>");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::header,
                           tokenType());
  OCCA_ASSERT_TRUE(token->to<occa::lang::headerToken>().systemHeader);
  setHeaderToken("\"foobar\"");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::header,
                           tokenType());
  OCCA_ASSERT_FALSE(token->to<occa::lang::headerToken>().systemHeader);
}
