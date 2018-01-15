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

#include "token.hpp"

void testSkipMethods();
void testPushPop();
void testPeekMethods();
void testTokenMethods();
void testStringMethods();
void testPrimitiveMethods();

std::string streamSource;
occa::lang::tokenStream stream(NULL);
occa::lang::token_t *token = NULL;

int main(const int argc, const char **argv) {
  testSkipMethods();
  testPushPop();
  testPeekMethods();
  testTokenMethods();
  testStringMethods();
  testPrimitiveMethods();
  if (token) {
    delete token;
  }
  return 0;
}

void setStream(const std::string &s) {
  streamSource = s;
  stream = occa::lang::tokenStream(NULL, streamSource.c_str());
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
  OCCA_ASSERT_EQUAL('b', *stream.fp.pos);
  stream.skipTo('e');
  OCCA_ASSERT_EQUAL('e', *stream.fp.pos);

  stream.fp.pos = c;
  stream.skipTo("c\n");
  OCCA_ASSERT_EQUAL(c + 1, stream.fp.pos);

  stream.fp.pos = c + 5;
  stream.skipFrom("\n");
  OCCA_ASSERT_EQUAL('d', *stream.fp.pos);
}

void testPushPop() {
  setStream("a\nb\nc\nd\ne");
  const char *c = streamSource.c_str();

  stream.push();
  stream.skipTo('c');
  OCCA_ASSERT_EQUAL(3,
                    stream.fp.line);
  OCCA_ASSERT_EQUAL(c + 4,
                    stream.fp.pos);
  stream.popAndRewind();
  OCCA_ASSERT_EQUAL(1,
                    stream.fp.line);
  OCCA_ASSERT_EQUAL(c + 0,
                    stream.fp.pos);
  stream.push();
  stream.push();
  stream.push();
  stream.skipTo('c');
  stream.pop();
  stream.pop();
  stream.pop();
  OCCA_ASSERT_EQUAL(3,
                    stream.fp.line);
  OCCA_ASSERT_EQUAL(c + 4,
                    stream.fp.pos);
}

#define testStringPeek(s, encoding_)                                    \
  setStream(s);                                                         \
  OCCA_ASSERT_EQUAL_BINARY((encoding_ << occa::lang::tokenType::encodingShift) | \
                           occa::lang::tokenType::string,               \
                           stream.peek())

#define testCharPeek(s, encoding_)                                      \
  setStream(s);                                                         \
  OCCA_ASSERT_EQUAL_BINARY((encoding_ << occa::lang::tokenType::encodingShift) | \
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

  setStream("@foo");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::attribute,
                           stream.peek());

  testStringPeek("\"\""                , 0);
  testStringPeek("\"string\\\"string\"", 0);
  testStringPeek("R\"(string)\""       , occa::lang::encodingType::R);
  testStringPeek("u8\"string\""        , occa::lang::encodingType::u8);
  testStringPeek("u\"string\""         , occa::lang::encodingType::u);
  testStringPeek("U\"string\""         , occa::lang::encodingType::U);
  testStringPeek("L\"string\""         , occa::lang::encodingType::L);
  testStringPeek("u8R\"(string)\""     , (occa::lang::encodingType::u8 |
                                          occa::lang::encodingType::R));
  testStringPeek("uR\"(string)\""      , (occa::lang::encodingType::u |
                                          occa::lang::encodingType::R));
  testStringPeek("UR\"(string)\""      , (occa::lang::encodingType::U |
                                          occa::lang::encodingType::R));
  testStringPeek("LR\"(string)\""      , (occa::lang::encodingType::L |
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
  setToken("@foo");
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::attribute,
                           tokenType());

  testStringToken("\"\""                , 0);
  testStringToken("\"string\\\"string\"", 0);
  testStringToken("R\"(string)\""       , occa::lang::encodingType::R);
  testStringToken("u8\"string\""        , occa::lang::encodingType::u8);
  testStringToken("u\"string\""         , occa::lang::encodingType::u);
  testStringToken("U\"string\""         , occa::lang::encodingType::U);
  testStringToken("L\"string\""         , occa::lang::encodingType::L);
  testStringToken("u8R\"(string)\""     , (occa::lang::encodingType::u8 |
                                           occa::lang::encodingType::R));
  testStringToken("uR\"(string)\""      , (occa::lang::encodingType::u |
                                           occa::lang::encodingType::R));
  testStringToken("UR\"(string)\""      , (occa::lang::encodingType::U |
                                           occa::lang::encodingType::R));
  testStringToken("LR\"(string)\""      , (occa::lang::encodingType::L |
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

#define testStringValue(s, value_)                              \
  setToken(s);                                                  \
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::string,       \
                           tokenType())                         \
  OCCA_ASSERT_EQUAL(value_,                                     \
                    token->to<occa::lang::stringToken>().value)


void addEncodingPrefixes(std::string &s1,
                         std::string &s2,
                         std::string &s3,
                         const std::string &encoding1,
                         const std::string &encoding2) {
  const std::string a  = "\"a\"";
  const std::string b  = "\"b\"";
  s1 = (encoding1 + a) + " " + b;
  s2 = a               + " " + (encoding2 + b);
  s3 = (encoding1 + a) + " " + (encoding2 + b);
}

void testStringMethods() {
  // Test values
  testStringValue("\"\""                , "");
  testStringValue("\"string\\\"string\"", "string\\\"string");
  testStringValue("R\"(string)\""       , "string");
  testStringValue("u8\"string\""        , "string");
  testStringValue("u\"string\""         , "string");
  testStringValue("U\"string\""         , "string");
  testStringValue("L\"string\""         , "string");
  testStringValue("u8R\"(string)\""     , "string");
  testStringValue("uR\"(string)\""      , "string");
  testStringValue("UR\"(string)\""      , "string");
  testStringValue("LR\"(string)\""      , "string");

  // Test raw strings
  testStringValue("R\"*(string)*\""  , "string");
  testStringValue("u8R\"*(string)*\"", "string");
  testStringValue("uR\"*(string)*\"" , "string");
  testStringValue("UR\"*(string)*\"" , "string");
  testStringValue("LR\"*(string)*\"" , "string");

  testStringValue("R\"foo(string)foo\""  , "string");
  testStringValue("u8R\"foo(string)foo\"", "string");
  testStringValue("uR\"foo(string)foo\"" , "string");
  testStringValue("UR\"foo(string)foo\"" , "string");
  testStringValue("LR\"foo(string)foo\"" , "string");

  // Test string concatination
  const std::string ab = "ab";
  std::string s1, s2, s3;
  addEncodingPrefixes(s1, s2, s3, "", "");
  testStringValue(s1, ab);
  testStringValue(s2, ab);
  testStringValue(s3, ab);
  testStringToken(s1, 0);
  testStringToken(s2, 0);
  testStringToken(s3, 0);

  addEncodingPrefixes(s1, s2, s3, "", "u");
  testStringValue(s1, ab);
  testStringValue(s2, ab);
  testStringValue(s3, ab);
  testStringToken(s1, 0);
  testStringToken(s2, occa::lang::encodingType::u);
  testStringToken(s3, occa::lang::encodingType::u);

  addEncodingPrefixes(s1, s2, s3, "", "U");
  testStringValue(s1, ab);
  testStringValue(s2, ab);
  testStringValue(s3, ab);
  testStringToken(s1, 0);
  testStringToken(s2, occa::lang::encodingType::U);
  testStringToken(s3, occa::lang::encodingType::U);

  addEncodingPrefixes(s1, s2, s3, "", "L");
  testStringValue(s1, ab);
  testStringValue(s2, ab);
  testStringValue(s3, ab);
  testStringToken(s1, 0);
  testStringToken(s2, occa::lang::encodingType::L);
  testStringToken(s3, occa::lang::encodingType::L);

  addEncodingPrefixes(s1, s2, s3, "u", "U");
  testStringValue(s1, ab);
  testStringValue(s2, ab);
  testStringValue(s3, ab);
  testStringToken(s1, occa::lang::encodingType::u);
  testStringToken(s2, occa::lang::encodingType::U);
  testStringToken(s3, occa::lang::encodingType::U);

  addEncodingPrefixes(s1, s2, s3, "u", "L");
  testStringValue(s1, ab);
  testStringValue(s2, ab);
  testStringValue(s3, ab);
  testStringToken(s1, occa::lang::encodingType::u);
  testStringToken(s2, occa::lang::encodingType::L);
  testStringToken(s3, occa::lang::encodingType::L);

  addEncodingPrefixes(s1, s2, s3, "U", "L");
  testStringValue(s1, ab);
  testStringValue(s2, ab);
  testStringValue(s3, ab);
  testStringToken(s1, occa::lang::encodingType::U);
  testStringToken(s2, occa::lang::encodingType::L);
  testStringToken(s3, occa::lang::encodingType::L);
}

#define testPrimitiveToken(type_, value_)                               \
  token = stream.getToken();                                            \
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::primitive,            \
                           tokenType());                                \
  OCCA_ASSERT_EQUAL(value_,                                             \
                    (type_) token->to<occa::lang::primitiveToken>().value)



void testPrimitiveMethods() {
  setStream("1 68719476735L +0.1 .1e-10 -4.5L");
  testPrimitiveToken(int, 1);
  testPrimitiveToken(int64_t, 68719476735L);
  testPrimitiveToken(float, +0.1);
  testPrimitiveToken(float, .1e-10);
  testPrimitiveToken(double, -4.5L);
}
