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
#include "./tokenUtils.hpp"

void testSkipMethods();
void testPushPop();
void testPeekMethods();
void testTokenMethods();
void testCommentSkipping();
void testStringMethods();
void testPrimitiveMethods();
void testErrors();

void setStream(const std::string &s) {
  static occa::lang::mergeStringTokens mergeStrings;

  tu::setStream(s);

  mergeStrings = occa::lang::mergeStringTokens();
  mergeStrings.addRef();
  stream.map(&mergeStrings);
}

void setToken(const std::string &s) {
  setStream(s);
  tu::getToken();
}

void setHeaderToken(const std::string &s) {
  setStream(s);
  if (token) {
    delete token;
  }
  token = stream.getHeaderToken();
}

int main(const int argc, const char **argv) {
  testSkipMethods();
  testPushPop();
  testPeekMethods();
  testTokenMethods();
  testCommentSkipping();
  testStringMethods();
  testPrimitiveMethods();
  testErrors();

  tu::free();

  return 0;
}

void testSkipMethods() {
  setStream("ab\nc\n\n\n\n\n\nd\ne");
  const char *c = streamSource.c_str();

  stream.skipTo('a');
  OCCA_ASSERT_EQUAL('a', *stream.fp.pos);

  stream.skipTo('b');
  OCCA_ASSERT_EQUAL('b', *stream.fp.pos);

  stream.skipTo('e');
  OCCA_ASSERT_EQUAL('e', *stream.fp.pos);

  stream.fp.pos = c;
  stream.skipTo("c\n");
  OCCA_ASSERT_EQUAL(c + 2, stream.fp.pos);

  stream.fp.pos = c + 6;
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

#define testStringPeek(s, encoding_)                      \
  setStream(s);                                           \
  OCCA_ASSERT_EQUAL_BINARY(                               \
    (encoding_ << occa::lang::tokenType::encodingShift) | \
    occa::lang::tokenType::string,                        \
    stream.peek())

#define testCharPeek(s, encoding_)                        \
  setStream(s);                                           \
  OCCA_ASSERT_EQUAL_BINARY(                               \
    (encoding_ << occa::lang::tokenType::encodingShift) | \
    occa::lang::tokenType::char_,                         \
    stream.peek())

void testPeekMethods() {
  setStream("abcd");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::identifier,
    stream.peek()
  );
  setStream("_abcd");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::identifier,
    stream.peek()
  );
  setStream("_abcd020230");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::identifier,
    stream.peek()
  );

  setStream("1");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::primitive,
    stream.peek()
  );
  setStream("2.0");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::primitive,
    stream.peek()
  );
  setStream("2.0e10");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::primitive,
    stream.peek()
  );
  setStream(".0e10-10");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::primitive,
    stream.peek()
  );

  setStream("+");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::op,
    stream.peek()
  );
  setStream("<");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::op,
    stream.peek()
  );
  setStream("->*");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::op,
    stream.peek()
  );
  setStream("==");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::op,
    stream.peek()
  );

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
                           tu::tokenType())                             \
  OCCA_ASSERT_EQUAL_BINARY(encoding_,                                   \
                           token->to<occa::lang::stringToken>().encoding)

#define testCharToken(s, encoding_)                                     \
  setToken(s);                                                          \
  OCCA_ASSERT_EQUAL_BINARY(occa::lang::tokenType::char_,                \
                           tu::tokenType())                             \
  OCCA_ASSERT_EQUAL_BINARY(encoding_,                                   \
                           token->to<occa::lang::charToken>().encoding)

void testTokenMethods() {
  setToken("abcd");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::identifier,
    tu::tokenType()
  );
  setToken("_abcd");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::identifier,
    tu::tokenType()
  );
  setToken("_abcd020230");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::identifier,
    tu::tokenType()
  );

  setToken("1");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::primitive,
    tu::tokenType()
  );
  setToken("2.0");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::primitive,
    tu::tokenType()
  );
  setToken("2.0e10");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::primitive,
    tu::tokenType()
  );
  setToken(".0e10-10");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::primitive,
    tu::tokenType()
  );

  setToken("+");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::op,
    tu::tokenType()
  );
  setToken("<");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::op,
    tu::tokenType()
  );
  setToken("->*");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::op,
    tu::tokenType()
  );
  setToken("==");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::op,
    tu::tokenType()
  );

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
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::header,
    tu::tokenType()
  );
  OCCA_ASSERT_TRUE(token->to<occa::lang::headerToken>().systemHeader);
  setHeaderToken("\"foobar\"");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::header,
    tu::tokenType()
  );
  OCCA_ASSERT_FALSE(token->to<occa::lang::headerToken>().systemHeader);
}

#define testStringValue(s, value_)              \
  setStream(s);                                 \
  testNextStringValue(value_)

#define testNextStringValue(value_)             \
  tu::getToken();                               \
  OCCA_ASSERT_EQUAL_BINARY(                     \
    occa::lang::tokenType::string,              \
    tu::tokenType()                             \
  );                                            \
  OCCA_ASSERT_EQUAL(                            \
    value_,                                     \
    token->to<occa::lang::stringToken>().value  \
  )


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

void testCommentSkipping() {
  setStream("// test    \n"
            "1.0");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::newline,
    stream.peek()
  );
  tu::getToken();
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::primitive,
    stream.peek()
  );
  setStream("/* 2.0 */ foo");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::identifier,
    stream.peek()
  );
  setStream("/*         \n"
            "2.0        \n"
            "*/ foo");
  OCCA_ASSERT_EQUAL_BINARY(
    occa::lang::tokenType::identifier,
    stream.peek()
  );
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

#define testPrimitiveToken(type_, value_)                 \
  tu::getToken();                                         \
  OCCA_ASSERT_EQUAL_BINARY(                               \
    occa::lang::tokenType::primitive,                     \
    tu::tokenType());                                     \
  OCCA_ASSERT_EQUAL(                                      \
    value_,                                               \
    (type_) token->to<occa::lang::primitiveToken>().value \
  )



void testPrimitiveMethods() {
  setStream("1 68719476735L +0.1 .1e-10 -4.5L");
  testPrimitiveToken(int, 1);
  testPrimitiveToken(int64_t, 68719476735L);
  tu::getToken();
  testPrimitiveToken(float, 0.1);
  testPrimitiveToken(float, .1e-10);
  tu::getToken();
  testPrimitiveToken(double, 4.5L);
}

void testErrors() {
  setStream("$ test\n\"foo\" \"bar\" ` bar foo");
  std::cerr << "Testing error outputs:\n";
  tu::getToken();
  stream.skipTo('\n');
  tu::getToken();
  testNextStringValue("foobar");
  tu::getToken();
}
